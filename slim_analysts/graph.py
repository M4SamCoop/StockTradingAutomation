"""
SlimTradingAgentsGraph — TradingAgents with the analyst phase replaced.

How it works
------------
TradingAgents' GraphSetup.setup_graph() calls:
    create_market_analyst(quick_llm)
    create_social_media_analyst(quick_llm)
    create_news_analyst(quick_llm)
    create_fundamentals_analyst(quick_llm)

Those names are resolved from the module namespace of tradingagents.graph.setup
(imported via `from tradingagents.agents import *`).  We monkey-patch those four
names in that module — just for the duration of TradingAgentsGraph.__init__ —
so when setup_graph() runs it picks up our slim factories instead.

Each slim factory closes over the TickerSnapshot dict so the node function can
look up pre-computed data by ticker at runtime.

Usage
-----
    from slim_analysts.graph import SlimTradingAgentsGraph

    graph = SlimTradingAgentsGraph(snapshots={"AAPL": snap}, config={...})
    result = graph.run("AAPL", "2026-06-11")
    print(result["action"])         # BUY / SELL / HOLD
    print(result["conviction"])     # 0.0 – 1.0
    print(result["raw_decision"])   # full final_trade_decision string
"""

import contextlib
import logging
import re
from typing import Any, Optional

import tradingagents.graph.setup as _ta_setup

from pipeline.models import TickerSnapshot
from slim_analysts.nodes import (
    make_slim_market_factory,
    make_slim_fundamentals_factory,
    make_slim_social_factory,
    make_slim_news_factory,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_SLIM_DEFAULT_CONFIG = {
    "llm_provider":           "anthropic",
    "deep_think_llm":         "claude-sonnet-4-6",
    "quick_think_llm":        "claude-haiku-4-5-20251001",
    "max_debate_rounds":      1,
    "max_risk_discuss_rounds": 1,
}


def _build_config(overrides: Optional[dict]) -> dict:
    """Merge user overrides on top of slim defaults + TradingAgents defaults."""
    try:
        from tradingagents.default_config import DEFAULT_CONFIG as _base
        cfg = dict(_base)
    except ImportError:
        cfg = {}
    cfg.update(_SLIM_DEFAULT_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Monkey-patch context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patch_analysts(snapshots: dict[str, TickerSnapshot]):
    """
    Temporarily replace the four analyst factory functions in
    tradingagents.graph.setup with slim versions that read from snapshots.

    Restores the originals on exit even if an exception is raised.
    """
    originals = {
        "create_market_analyst":        getattr(_ta_setup, "create_market_analyst",        None),
        "create_fundamentals_analyst":  getattr(_ta_setup, "create_fundamentals_analyst",  None),
        "create_social_media_analyst":  getattr(_ta_setup, "create_social_media_analyst",  None),
        "create_news_analyst":          getattr(_ta_setup, "create_news_analyst",           None),
    }

    _ta_setup.create_market_analyst       = make_slim_market_factory(snapshots)
    _ta_setup.create_fundamentals_analyst = make_slim_fundamentals_factory(snapshots)
    _ta_setup.create_social_media_analyst = make_slim_social_factory(snapshots)
    _ta_setup.create_news_analyst         = make_slim_news_factory(snapshots)

    try:
        yield
    finally:
        for name, fn in originals.items():
            if fn is not None:
                setattr(_ta_setup, name, fn)
            else:
                try:
                    delattr(_ta_setup, name)
                except AttributeError:
                    pass


# ---------------------------------------------------------------------------
# SlimTradingAgentsGraph
# ---------------------------------------------------------------------------

class SlimTradingAgentsGraph:
    """
    Wraps TradingAgentsGraph with the 4 analyst nodes replaced by local
    pre-computed data from TickerSnapshot.

    Parameters
    ----------
    snapshots : dict[str, TickerSnapshot]
        Keyed by ticker symbol (upper-case).  The slim nodes look up the
        correct snapshot at runtime from state["company_of_interest"].
    config : dict, optional
        Overrides for TradingAgents config.  Defaults to Anthropic / Haiku+Sonnet.
    selected_analysts : list[str], optional
        Which analysts to include.  Defaults to all four.
    """

    def __init__(
        self,
        snapshots: dict[str, TickerSnapshot],
        config: Optional[dict] = None,
        selected_analysts: Optional[list] = None,
    ):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        self._snapshots = {k.upper(): v for k, v in snapshots.items()}
        self._config = _build_config(config)
        self._selected_analysts = selected_analysts or ["market", "social", "news", "fundamentals"]

        logger.info(
            "Initialising SlimTradingAgentsGraph "
            f"(tickers={list(self._snapshots)}, analysts={self._selected_analysts})"
        )

        with _patch_analysts(self._snapshots):
            self._ta_graph = TradingAgentsGraph(
                selected_analysts=self._selected_analysts,
                config=self._config,
                debug=False,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, ticker: str, date: str) -> dict:
        """
        Run the full TradingAgents reasoning chain for one ticker.

        Returns
        -------
        dict with keys:
            action      : "BUY" | "SELL" | "HOLD"
            conviction  : float 0.0 – 1.0
            raw_decision: full final_trade_decision string
            judge_decision: investment debate judge text
            state       : the complete final AgentState dict
        """
        ticker = ticker.upper()

        if ticker not in self._snapshots:
            raise ValueError(
                f"Ticker '{ticker}' not in snapshots. "
                f"Available: {list(self._snapshots)}"
            )

        logger.info(f"Running reasoning chain for {ticker} on {date}")

        final_state, processed_signal = self._ta_graph.propagate(ticker, date)

        action = _extract_action(processed_signal or "")
        conviction = _extract_conviction(
            final_state.get("investment_debate_state", {}).get("judge_decision", ""),
            action,
        )

        return {
            "action":       action,
            "conviction":   conviction,
            "raw_decision": final_state.get("final_trade_decision", ""),
            "judge_decision": final_state.get("investment_debate_state", {}).get("judge_decision", ""),
            "market_report":       final_state.get("market_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", ""),
            "sentiment_report":    final_state.get("sentiment_report", ""),
            "news_report":         final_state.get("news_report", ""),
            "state": final_state,
        }


# ---------------------------------------------------------------------------
# Signal parsing helpers
# ---------------------------------------------------------------------------

_ACTION_PATTERNS = [
    (re.compile(r"\bSTRONG\s+BUY\b",  re.I), "BUY"),
    (re.compile(r"\bSTRONG\s+SELL\b", re.I), "SELL"),
    (re.compile(r"\b(?:FINAL\s+)?(?:DECISION|RECOMMENDATION|SIGNAL)\s*[:\-–]\s*(BUY|SELL|HOLD)\b", re.I), None),  # capture group
    (re.compile(r"\b(BUY|SELL|HOLD)\b", re.I), None),  # capture group fallback
]

def _extract_action(text: str) -> str:
    """Extract BUY / SELL / HOLD from processed signal text."""
    for pat, fixed in _ACTION_PATTERNS:
        m = pat.search(text)
        if m:
            if fixed:
                return fixed
            val = m.group(1).upper()
            return val if val in ("BUY", "SELL", "HOLD") else "HOLD"
    return "HOLD"


_CONVICTION_KEYWORDS = {
    "strong":    0.90,
    "high":      0.80,
    "moderate":  0.60,
    "cautious":  0.50,
    "weak":      0.35,
    "minimal":   0.25,
    "uncertain": 0.20,
    "mixed":     0.45,
    "split":     0.45,
    "neutral":   0.50,
    "compelling": 0.80,
    "clear":     0.75,
    "solid":     0.70,
}

def _extract_conviction(judge_text: str, action: str) -> float:
    """
    Heuristic conviction score 0.0–1.0 from the judge's decision text.

    Strategy:
    1. Look for explicit percent or numeric confidence mentions.
    2. Match conviction-level keywords.
    3. Fall back to 0.5 if nothing found.
    4. If action is HOLD, cap at 0.6 (holds are inherently uncertain).
    """
    if not judge_text:
        return 0.5

    text_lower = judge_text.lower()

    # Explicit percentage: "70% confident", "confidence: 80%"
    pct_match = re.search(r"(\d{1,3})\s*%\s*(?:confident|confidence|probability|probability)", text_lower)
    if pct_match:
        return min(1.0, max(0.0, int(pct_match.group(1)) / 100.0))

    # Explicit score like "conviction 8/10" or "8 out of 10"
    score_match = re.search(r"(\d+)\s*/\s*10", text_lower)
    if score_match:
        return min(1.0, int(score_match.group(1)) / 10.0)

    # Keyword scan (first hit wins)
    for kw, score in _CONVICTION_KEYWORDS.items():
        if kw in text_lower:
            conviction = score
            if action == "HOLD":
                conviction = min(conviction, 0.6)
            return conviction

    return 0.5
