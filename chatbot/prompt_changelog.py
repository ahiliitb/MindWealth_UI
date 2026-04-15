"""
Lightweight changelog for prompt iterations.

Tracks what changed in each named prompt, when, and why.
Quality scores can be attached to any version so you can correlate
prompt iterations against report quality over time.

Stored in: chatbot/prompt_changelog.json

Typical usage
-------------
    changelog = PromptChangelog()

    # Auto-detect changes on every engine start-up:
    changelog.auto_register("SYSTEM_PROMPT", SYSTEM_PROMPT, reason="tightened hallucination guard")

    # After reviewing a report, rate the current prompt version:
    changelog.record_quality("SYSTEM_PROMPT", score=4.5, notes="Good breadth coverage, verbose intro")

    # See the full version history (metadata only, no giant prompt bodies):
    changelog.get_history("SYSTEM_PROMPT")

    # See what exactly changed between two versions:
    print(changelog.diff("SYSTEM_PROMPT", -2, -1))

    # See quality trends per version:
    changelog.quality_report("SYSTEM_PROMPT")
"""

import difflib
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CHATBOT_DIR = Path(__file__).resolve().parent
CHANGELOG_FILE = _CHATBOT_DIR / "prompt_changelog.json"


class PromptChangelog:
    """
    Version-tracks named prompts with reason annotations and quality scores.

    Change detection is hash-based (MD5 of full content), so registering
    the same prompt string twice is a no-op — no duplicate entries are
    created even when the engine restarts repeatedly.

    Version numbering is 1-based and monotonically increasing per prompt.
    """

    def __init__(self) -> None:
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    # ── persistence ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if CHANGELOG_FILE.exists():
            try:
                with open(CHANGELOG_FILE, "r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
            except Exception as exc:
                logger.error(f"Failed to load prompt changelog: {exc}")
                self._data = {}

    def _save(self) -> None:
        try:
            with open(CHANGELOG_FILE, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"Failed to save prompt changelog: {exc}")

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]

    def _versions(self, name: str) -> List[Dict[str, Any]]:
        """Return (and lazily create) the version list for *name*."""
        return self._data.setdefault(name, [])

    # ── write ────────────────────────────────────────────────────────────────

    def auto_register(
        self,
        name: str,
        content: str,
        reason: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Register *content* as the latest version of prompt *name*.

        Returns the new version entry if the content has changed since the
        last recorded version, or *None* if it is unchanged (idempotent).

        Parameters
        ----------
        name:
            Logical prompt identifier, e.g. ``"SYSTEM_PROMPT"``.
        content:
            The full prompt string.
        reason:
            Human-readable explanation for the change, e.g.
            ``"tightened hallucination guard for breadth signals"``.
        """
        versions = self._versions(name)
        new_hash = self._hash(content)

        if versions and versions[-1]["hash"] == new_hash:
            return None  # no change — skip

        version_number = len(versions) + 1
        entry: Dict[str, Any] = {
            "version": version_number,
            "timestamp": datetime.now().isoformat(),
            "hash": new_hash,
            "reason": reason or "updated",
            # First 300 chars — enough to quickly orient a human reviewer
            "snapshot": content[:300],
            "full_content": content,
            "quality_scores": [],
        }
        versions.append(entry)
        self._save()
        logger.info(f"Prompt '{name}' logged as v{version_number} (hash={new_hash})")
        return entry

    def record_quality(
        self,
        name: str,
        score: float,
        notes: str = "",
        version: Optional[int] = None,
    ) -> bool:
        """
        Attach a quality rating to a prompt version.

        Parameters
        ----------
        name:
            Prompt identifier.
        score:
            Numeric quality score (suggested scale 1–5, but any range works).
        notes:
            Free-text annotation, e.g. ``"hallucinated 1 ticker name"``.
        version:
            1-based version number.  Defaults to the latest version.

        Returns
        -------
        bool
            True on success, False if the prompt / version was not found.
        """
        versions = self._versions(name)
        if not versions:
            logger.warning(f"No versions found for prompt '{name}'")
            return False

        # Convert to 0-based index; default = last entry
        idx = (version - 1) if version is not None else len(versions) - 1
        if idx < 0 or idx >= len(versions):
            logger.warning(f"Version {version} out of range for prompt '{name}'")
            return False

        versions[idx]["quality_scores"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "score": score,
                "notes": notes,
            }
        )
        self._save()
        return True

    # ── read ─────────────────────────────────────────────────────────────────

    def get_history(
        self, name: str, include_full_content: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Return all recorded versions for *name*, oldest first.

        Set *include_full_content=True* to include the complete prompt body
        in each entry (omitted by default to keep output readable).
        """
        versions = self._versions(name)
        if include_full_content:
            return list(versions)
        return [
            {k: v for k, v in entry.items() if k != "full_content"}
            for entry in versions
        ]

    def list_prompts(self) -> List[str]:
        """Return the names of all tracked prompts."""
        return list(self._data.keys())

    def current_version(self, name: str) -> Optional[Dict[str, Any]]:
        """Return metadata for the latest version of *name* (no full_content)."""
        versions = self._versions(name)
        if not versions:
            return None
        entry = versions[-1]
        return {k: v for k, v in entry.items() if k != "full_content"}

    def diff(self, name: str, v1: int = -2, v2: int = -1) -> str:
        """
        Unified diff between two versions identified by 0-based index.

        Defaults to comparing the *last two* versions (``v1=-2, v2=-1``).
        Pass explicit 0-based indices to compare any pair.

        Returns a human-readable diff string, or an explanatory message
        if fewer than two versions exist.
        """
        versions = self._versions(name)
        if len(versions) < 2:
            return f"Only {len(versions)} version(s) recorded for '{name}' — nothing to diff yet."

        try:
            entry_a = versions[v1]
            entry_b = versions[v2]
        except IndexError as exc:
            return f"Index error: {exc}"

        a_lines = entry_a["full_content"].splitlines(keepends=True)
        b_lines = entry_b["full_content"].splitlines(keepends=True)
        label_a = f"v{entry_a['version']}  {entry_a['timestamp'][:10]}  [{entry_a['reason']}]"
        label_b = f"v{entry_b['version']}  {entry_b['timestamp'][:10]}  [{entry_b['reason']}]"

        result = "".join(difflib.unified_diff(a_lines, b_lines, fromfile=label_a, tofile=label_b))
        return result if result else "No textual differences found between the selected versions."

    def quality_report(self, name: str) -> Dict[str, Any]:
        """
        Return a per-version quality summary for *name*.

        Each version entry shows:
        - ``avg_quality``: mean score (None if unrated)
        - ``num_ratings``: how many ratings have been recorded
        - ``ratings``: the individual score entries

        Use this to correlate prompt changes against report quality over time.
        """
        versions = self._versions(name)
        version_summaries: List[Dict[str, Any]] = []

        for v in versions:
            scores = v.get("quality_scores", [])
            avg = (
                round(sum(s["score"] for s in scores) / len(scores), 2)
                if scores
                else None
            )
            version_summaries.append(
                {
                    "version": v["version"],
                    "date": v["timestamp"][:10],
                    "hash": v["hash"],
                    "reason": v["reason"],
                    "snapshot": v.get("snapshot", "")[:120],
                    "avg_quality": avg,
                    "num_ratings": len(scores),
                    "ratings": scores,
                }
            )

        return {"prompt": name, "versions": version_summaries}

    def summary(self) -> str:
        """Return a human-readable one-liner per prompt for quick status checks."""
        if not self._data:
            return "No prompts tracked yet."
        lines: List[str] = []
        for name, versions in self._data.items():
            if not versions:
                continue
            latest = versions[-1]
            scores = latest.get("quality_scores", [])
            avg = (
                f"{sum(s['score'] for s in scores)/len(scores):.1f}"
                if scores
                else "unrated"
            )
            lines.append(
                f"  {name}: {len(versions)} version(s), "
                f"latest v{latest['version']} on {latest['timestamp'][:10]} "
                f"[{latest['reason']}], quality={avg}"
            )
        return "Prompt Changelog:\n" + "\n".join(lines)
