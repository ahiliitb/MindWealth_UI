"""
Parallel Orchestrator for MindWealth Chatbot — HYBRID route.

Fans out web search and internal signal data fetch concurrently using
ThreadPoolExecutor, then collects results with independent timeouts so
a slow/failing branch never blocks the other.

Usage
-----
orchestrator = ParallelOrchestrator()
result = orchestrator.run(
    web_fn=lambda: web_agent.run(user_message, queries, context),
    internal_fn=lambda: engine._fetch_signal_data(...),
    web_timeout=12,
    internal_timeout=45,
)
# result.web_result    — WebSearchResult or None
# result.signal_data   — fetched_data dict or None
# result.signal_metadata — extraction metadata dict or None
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Aggregated result from both parallel sub-agents."""

    web_result: Optional[Any]           # WebSearchResult | None
    signal_data: Optional[Dict]         # fetched_data dict from _fetch_signal_data()
    signal_metadata: Optional[Dict]     # extraction metadata (columns, signal_types, etc.)
    web_failed: bool = False            # True if web branch timed out or raised
    internal_failed: bool = False       # True if internal branch timed out or raised
    web_error: Optional[str] = None     # Error message if web_failed
    internal_error: Optional[str] = None
    elapsed_ms: float = 0.0


class ParallelOrchestrator:
    """
    Runs two callables concurrently and collects results with independent timeouts.

    Both branches are always attempted.  A timeout or exception in one branch
    sets the corresponding ``*_failed`` flag on the result — the other branch's
    output is still used.  This ensures graceful degradation: if web search is
    slow, the internal signal answer still reaches the user (and vice-versa).
    """

    def run(
        self,
        web_fn: Callable[[], Any],
        internal_fn: Callable[[], Tuple[Dict, Dict]],
        web_timeout: float = 12.0,
        internal_timeout: float = 45.0,
    ) -> OrchestratorResult:
        """
        Execute web_fn and internal_fn in parallel.

        Parameters
        ----------
        web_fn:
            Callable that returns a WebSearchResult.  Should not raise; any
            exception is caught and recorded in ``result.web_error``.
        internal_fn:
            Callable that returns (fetched_data: Dict, extraction_meta: Dict).
        web_timeout:
            Seconds to wait for the web branch before giving up.
        internal_timeout:
            Seconds to wait for the internal branch before giving up.

        Returns
        -------
        OrchestratorResult
        """
        t0 = time.monotonic()

        web_future: Optional[Future] = None
        internal_future: Optional[Future] = None

        web_result = None
        signal_data = None
        signal_metadata = None
        web_failed = False
        internal_failed = False
        web_error = None
        internal_error = None

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid_agent") as executor:
            # Submit both tasks immediately — they start in parallel.
            logger.info(
                f"[FLOW 3/7] Orchestrator spawning parallel agents  |  "
                f"web_timeout={web_timeout}s  internal_timeout={internal_timeout}s"
            )
            web_future = executor.submit(web_fn)
            logger.info("[FLOW 3/7] Thread-A (WebAgent) started")
            internal_future = executor.submit(internal_fn)
            logger.info("[FLOW 3/7] Thread-B (InternalAgent) started")

            # Collect web result with its own timeout.
            logger.info("[FLOW 4/7] Waiting for Thread-A (WebAgent) result ...")
            try:
                web_result = web_future.result(timeout=web_timeout)
                web_sources_count = len(getattr(web_result, "results", []) or [])
                logger.info(
                    f"[FLOW 4/7 | Thread-A] WebAgent completed  |  "
                    f"success={getattr(web_result, 'success', web_result is not None)}  "
                    f"results={web_sources_count}"
                )
            except TimeoutError:
                web_failed = True
                web_error = f"Web search timed out after {web_timeout}s"
                logger.warning(f"[FLOW 4/7 | Thread-A] WebAgent TIMED OUT — {web_error}")
                web_future.cancel()
            except Exception as exc:
                web_failed = True
                web_error = str(exc)
                logger.warning(f"[FLOW 4/7 | Thread-A] WebAgent FAILED — {exc}")

            # Collect internal result with its own timeout.
            logger.info("[FLOW 4/7] Waiting for Thread-B (InternalAgent) result ...")
            try:
                internal_result = internal_future.result(timeout=internal_timeout)
                if isinstance(internal_result, tuple) and len(internal_result) == 2:
                    signal_data, signal_metadata = internal_result
                else:
                    signal_data = internal_result
                    signal_metadata = {}
                total_rows = sum(
                    len(v) for v in (signal_data or {}).values() if hasattr(v, "__len__")
                )
                signal_types_fetched = list((signal_data or {}).keys())
                logger.info(
                    f"[FLOW 4/7 | Thread-B] InternalAgent completed  |  "
                    f"signal_types={signal_types_fetched}  total_rows={total_rows}"
                )
            except TimeoutError:
                internal_failed = True
                internal_error = f"Internal data fetch timed out after {internal_timeout}s"
                logger.warning(f"[FLOW 4/7 | Thread-B] InternalAgent TIMED OUT — {internal_error}")
                internal_future.cancel()
            except Exception as exc:
                internal_failed = True
                internal_error = str(exc)
                logger.error(f"[FLOW 4/7 | Thread-B] InternalAgent FAILED — {exc}")

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            f"[FLOW 4/7] Both agents finished  |  "
            f"elapsed={elapsed_ms:.0f}ms  "
            f"web={'OK' if not web_failed else 'FAILED'}  "
            f"internal={'OK' if not internal_failed else 'FAILED'}"
        )

        return OrchestratorResult(
            web_result=web_result,
            signal_data=signal_data,
            signal_metadata=signal_metadata,
            web_failed=web_failed,
            internal_failed=internal_failed,
            web_error=web_error,
            internal_error=internal_error,
            elapsed_ms=elapsed_ms,
        )
