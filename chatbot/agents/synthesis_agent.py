"""
Synthesis Agent — builds a structured, source-separated prompt for Claude
when both internal signal data and live web search results are available
(HYBRID route).

The prompt explicitly labels each source, gives Claude reconciliation rules,
and signals which branches succeeded or failed so Claude never speculates
about missing data.

Usage
-----
synthesis = SynthesisAgent()
prompt = synthesis.build_prompt(
    user_message=user_message,
    web_result=orch_result.web_result,       # WebSearchResult | None
    signal_data=orch_result.signal_data,     # fetched_data dict | None
    signal_metadata=orch_result.signal_metadata,  # extraction meta | None
    web_failed=orch_result.web_failed,
    internal_failed=orch_result.internal_failed,
)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum characters of formatted web context to inject (keeps prompt bounded)
_MAX_WEB_CHARS = 4000
# Maximum characters of signal JSON to inject
_MAX_SIGNAL_CHARS = 8000


class SynthesisAgent:
    """
    Constructs a structured synthesis prompt for Claude that separates
    internal signal data (SOURCE A) from live web context (SOURCE B) and
    provides explicit reconciliation instructions.

    Design goals
    ------------
    - Claude always knows which source each piece of information came from.
    - If a source failed or timed out, Claude is told not to speculate.
    - If SOURCE B (web) contradicts SOURCE A (signals), Claude must surface
      the conflict rather than silently override signal values.
    - Prompt length is bounded so we don't blow the context window.
    """

    def build_prompt(
        self,
        user_message: str,
        web_result: Optional[Any],          # WebSearchResult | None
        signal_data: Optional[Dict],        # fetched_data: {signal_type: DataFrame}
        signal_metadata: Optional[Dict],    # extraction metadata dict
        web_failed: bool = False,
        internal_failed: bool = False,
        web_error: Optional[str] = None,
        internal_error: Optional[str] = None,
    ) -> str:
        """
        Build and return the synthesized prompt string.

        Parameters
        ----------
        user_message:
            The original user question (unmodified).
        web_result:
            WebSearchResult from the web agent, or None.
        signal_data:
            Dict mapping signal_type → DataFrame, or None.
        signal_metadata:
            Extraction metadata dict (columns_by_signal_type, reasoning, …).
        web_failed / internal_failed:
            Flags from OrchestratorResult.
        web_error / internal_error:
            Error messages for failed branches.

        Returns
        -------
        str
            A fully structured prompt ready to send to Claude.
        """
        now_iso = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info(
            "[FLOW 6/7] SynthesisAgent.build_prompt() started  |  "
            f"web_failed={web_failed}  internal_failed={internal_failed}  "
            f"has_web={'yes' if web_result is not None else 'no'}  "
            f"has_signal_data={'yes' if signal_data else 'no'}"
        )

        source_a_block = self._build_source_a(
            signal_data, signal_metadata, internal_failed, internal_error, now_iso
        )
        source_b_block = self._build_source_b(
            web_result, web_failed, web_error, now_iso
        )
        status_summary = self._build_status_summary(
            signal_data, web_result, web_failed, internal_failed
        )
        instructions = self._build_instructions()

        parts = [
            f"User Query: {user_message}",
            "",
            source_a_block,
            "",
            source_b_block,
            "",
            "=== SYNTHESIS INSTRUCTIONS ===",
            instructions,
            "",
            "=== SOURCE STATUS ===",
            status_summary,
        ]
        prompt = "\n".join(parts)

        logger.info(
            f"[FLOW 6/7] SynthesisAgent prompt ready  |  "
            f"total_chars={len(prompt)}  "
            f"source_a_chars={len(source_a_block)}  "
            f"source_b_chars={len(source_b_block)}"
        )
        return prompt

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_source_a(
        self,
        signal_data: Optional[Dict],
        signal_metadata: Optional[Dict],
        failed: bool,
        error: Optional[str],
        now_iso: str,
    ) -> str:
        lines = [
            "=== SOURCE A: MINDWEALTH SIGNAL DATA (internal — primary source) ===",
        ]

        if failed:
            lines.append(f"STATUS: FAILED — {error or 'Internal data fetch error'}")
            lines.append("Do NOT speculate about signal data. State it is unavailable.")
            return "\n".join(lines)

        if not signal_data:
            lines.append("STATUS: NO DATA RETURNED")
            lines.append("No signal data was fetched for this query.")
            return "\n".join(lines)

        meta = signal_metadata or {}
        columns_by_st = meta.get("columns_by_signal_type", {})
        reasoning_by_st = meta.get("reasoning_by_signal_type", {})

        total_rows = 0
        for st, df in signal_data.items():
            try:
                total_rows += len(df)
            except Exception:
                pass

        lines.append(f"Fetched at: {now_iso} | Total rows: {total_rows}")
        lines.append("")

        for signal_type, df in signal_data.items():
            try:
                row_count = len(df)
            except Exception:
                row_count = "?"
            cols = columns_by_st.get(signal_type, [])
            reasoning = reasoning_by_st.get(signal_type, "")

            lines.append(f"--- Signal Type: {signal_type.upper()} | Rows: {row_count} ---")
            if cols:
                lines.append(f"Columns selected: {', '.join(cols)}")
            if reasoning:
                lines.append(f"Column reasoning: {reasoning[:120]}")

            try:
                records = df.to_dict("records")
                payload = {
                    "signal_type": signal_type,
                    "record_count": row_count,
                    "columns_selected": cols,
                    "data": records,
                }
                json_str = json.dumps(payload, indent=2, default=str)
                if len(json_str) > _MAX_SIGNAL_CHARS:
                    json_str = json_str[:_MAX_SIGNAL_CHARS] + "\n... [truncated for prompt length]"
                lines.append(json_str)
            except Exception as exc:
                lines.append(f"[Could not serialise signal data: {exc}]")

        return "\n".join(lines)

    def _build_source_b(
        self,
        web_result: Optional[Any],
        failed: bool,
        error: Optional[str],
        now_iso: str,
    ) -> str:
        lines = [
            "=== SOURCE B: LIVE WEB CONTEXT (supplementary — use to enrich SOURCE A) ===",
        ]

        if failed:
            lines.append(f"STATUS: FAILED — {error or 'Web search timed out or errored'}")
            lines.append("Do NOT speculate about web content. Answer from SOURCE A only.")
            return "\n".join(lines)

        if web_result is None:
            lines.append("STATUS: SKIPPED — Web search was not executed for this query.")
            return "\n".join(lines)

        if not getattr(web_result, "success", False):
            err = getattr(web_result, "error", "unknown error")
            lines.append(f"STATUS: FAILED — {err}")
            lines.append("Do NOT speculate about web content. Answer from SOURCE A only.")
            return "\n".join(lines)

        queries_used = getattr(web_result, "search_queries_used", [])
        sources = getattr(web_result, "sources", [])
        formatted = getattr(web_result, "formatted_context", "")

        lines.append(f"Retrieved at: {now_iso}")
        if queries_used:
            lines.append(f"Search queries used: {queries_used}")
        if sources:
            lines.append(f"Sources ({len(sources)}): {sources}")
        lines.append("")

        if len(formatted) > _MAX_WEB_CHARS:
            formatted = formatted[:_MAX_WEB_CHARS] + "\n... [web context truncated for prompt length]"
        lines.append(formatted)

        return "\n".join(lines)

    @staticmethod
    def _build_instructions() -> str:
        return (
            "1. Answer the user's question using SOURCE A (MindWealth signal data) as the PRIMARY source.\n"
            "2. Use SOURCE B (live web context) ONLY to enrich, contextualise, or add recency to SOURCE A findings.\n"
            "3. If SOURCE B contradicts SOURCE A values (prices, dates, Sharpe ratios, function names), "
            "surface the conflict explicitly — do NOT silently override SOURCE A values with web data.\n"
            "4. If a source is marked FAILED or SKIPPED, do not speculate about its content — "
            "clearly note that information was unavailable.\n"
            "5. Always cite web sources with [Source N] tags where applicable.\n"
            "6. Keep the response concise and proportional to the question — avoid padding."
        )

    @staticmethod
    def _build_status_summary(
        signal_data: Optional[Dict],
        web_result: Optional[Any],
        web_failed: bool,
        internal_failed: bool,
    ) -> str:
        # Signal data status
        if internal_failed:
            sig_status = "FAILED"
        elif signal_data:
            total = sum(
                (len(df) if hasattr(df, "__len__") else 0) for df in signal_data.values()
            )
            sig_status = f"OK ({total} rows across {len(signal_data)} signal type(s))"
        else:
            sig_status = "NO DATA"

        # Web status
        if web_failed:
            web_status = "FAILED (timeout or error)"
        elif web_result is None:
            web_status = "SKIPPED"
        elif getattr(web_result, "success", False):
            n = len(getattr(web_result, "results", []))
            web_status = f"OK ({n} result(s))"
        else:
            web_status = f"FAILED ({getattr(web_result, 'error', 'unknown')})"

        return f"  Signal data (SOURCE A): {sig_status}\n  Web search  (SOURCE B): {web_status}"
