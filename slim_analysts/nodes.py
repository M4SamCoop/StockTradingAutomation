"""
Slim analyst node factories.

Each function returns a factory with signature `(llm) -> node_fn` — the same
signature TradingAgents' GraphSetup expects for create_market_analyst etc.

The llm argument is accepted but ignored; the node returns a pre-built report
directly so no LLM tool calls happen in the analyst phase at all.

The node returns:
  {"messages": [AIMessage(content=report)], "<type>_report": report}

Because the AIMessage has no tool_calls, TradingAgents' conditional_logic
routes straight to "Msg Clear <Type>" — bypassing the ToolNode entirely.
"""

from typing import Optional
from langchain_core.messages import AIMessage

from pipeline.models import TickerSnapshot
from slim_analysts.formatters import (
    format_technical_report,
    format_fundamentals_report,
    format_sentiment_report,
    format_news_report,
)


# ---------------------------------------------------------------------------
# Factory makers — each takes a snapshot registry and returns a factory fn
# ---------------------------------------------------------------------------

def make_slim_market_factory(snapshots: dict[str, TickerSnapshot]):
    """
    Returns create_slim_market_analyst(llm) -> node_fn.
    Drop-in replacement for tradingagents.agents.create_market_analyst.
    """
    def create_slim_market_analyst(llm):
        def market_analyst_node(state):
            ticker = state["company_of_interest"]
            snap = snapshots.get(ticker)
            report = format_technical_report(snap) if snap else (
                f"TECHNICAL SUMMARY — {ticker}\n\nNo snapshot available."
            )
            return {
                "messages": [AIMessage(content=report)],
                "market_report": report,
            }
        return market_analyst_node
    return create_slim_market_analyst


def make_slim_fundamentals_factory(snapshots: dict[str, TickerSnapshot]):
    """
    Returns create_slim_fundamentals_analyst(llm) -> node_fn.
    Drop-in for tradingagents.agents.create_fundamentals_analyst.
    """
    def create_slim_fundamentals_analyst(llm):
        def fundamentals_analyst_node(state):
            ticker = state["company_of_interest"]
            snap = snapshots.get(ticker)
            report = format_fundamentals_report(snap) if snap else (
                f"FUNDAMENTALS SUMMARY — {ticker}\n\nNo snapshot available."
            )
            return {
                "messages": [AIMessage(content=report)],
                "fundamentals_report": report,
            }
        return fundamentals_analyst_node
    return create_slim_fundamentals_analyst


def make_slim_social_factory(snapshots: dict[str, TickerSnapshot]):
    """
    Returns create_slim_social_media_analyst(llm) -> node_fn.
    Drop-in for tradingagents.agents.create_social_media_analyst.
    """
    def create_slim_social_media_analyst(llm):
        def social_analyst_node(state):
            ticker = state["company_of_interest"]
            snap = snapshots.get(ticker)
            report = format_sentiment_report(snap) if snap else (
                f"SENTIMENT & SOCIAL SUMMARY — {ticker}\n\nNo snapshot available."
            )
            return {
                "messages": [AIMessage(content=report)],
                "sentiment_report": report,
            }
        return social_analyst_node
    return create_slim_social_media_analyst


def make_slim_news_factory(snapshots: dict[str, TickerSnapshot]):
    """
    Returns create_slim_news_analyst(llm) -> node_fn.
    Drop-in for tradingagents.agents.create_news_analyst.
    """
    def create_slim_news_analyst(llm):
        def news_analyst_node(state):
            ticker = state["company_of_interest"]
            snap = snapshots.get(ticker)
            report = format_news_report(snap) if snap else (
                f"NEWS SUMMARY — {ticker}\n\nNo snapshot available."
            )
            return {
                "messages": [AIMessage(content=report)],
                "news_report": report,
            }
        return news_analyst_node
    return create_slim_news_analyst
