"""
Phase 3 runner — ties the local data pipeline (Phases 1+2) to the
TradingAgents reasoning layer (Phase 3).

Usage
-----
    python3 phase3_runner.py --tickers AAPL MSFT --date 2026-06-11
    python3 phase3_runner.py --tickers AAPL --date 2026-06-11 --no-fundamentals

Flow
----
1. Screen the watchlist (or use --tickers directly).
2. Assemble TickerSnapshot objects (technical + fundamentals + sentiment).
3. Pass snapshots to SlimTradingAgentsGraph.
4. Run one ticker at a time through the reasoning chain.
5. Print structured decision + conviction for each ticker.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Phase 3: run TradingAgents reasoning on pre-assembled snapshots."
    )
    p.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Tickers to analyse. Skips the screener.",
    )
    p.add_argument(
        "--date", default=None,
        help="Analysis date YYYY-MM-DD (default: today).",
    )
    p.add_argument(
        "--no-fundamentals", action="store_true",
        help="Skip fundamental data (faster for testing).",
    )
    p.add_argument(
        "--no-sentiment", action="store_true",
        help="Skip FinBERT sentiment (if transformers not installed).",
    )
    p.add_argument(
        "--analysts", nargs="+",
        default=["market", "social", "news", "fundamentals"],
        metavar="ANALYST",
        help="Subset of analysts to activate (default: all four).",
    )
    p.add_argument(
        "--output", default=None, metavar="FILE",
        help="Write decisions to a JSON file.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full reports and debate output.",
    )
    p.add_argument(
        "--screen", action="store_true",
        help="Run the screener first; ignore --tickers.",
    )
    p.add_argument(
        "--shortlist-size", type=int, default=3,
        help="Number of tickers to select from screener (default: 3).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    date = args.date or datetime.today().strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Step 1: Determine tickers
    # ------------------------------------------------------------------
    if args.screen:
        print("Running screener…", flush=True)
        from pipeline.screener import score_tickers
        from config.watchlist import SP500_HIGH_LIQUIDITY
        scored = score_tickers(SP500_HIGH_LIQUIDITY, date=date, max_workers=10)
        tickers = [s.ticker for s in scored[: args.shortlist_size]]
        print(f"Screener selected: {tickers}\n")
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        print("Error: provide --tickers AAPL MSFT … or use --screen.", file=sys.stderr)
        sys.exit(1)

    if not tickers:
        print("No tickers to analyse.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Assemble snapshots
    # ------------------------------------------------------------------
    print(f"Assembling snapshots for {tickers} on {date}…", flush=True)
    from pipeline.data_assembler import assemble_snapshots

    snapshots_list = assemble_snapshots(
        tickers,
        date=date,
        include_fundamentals=not args.no_fundamentals,
        include_sentiment=not args.no_sentiment,
    )
    snapshots = {s.ticker: s for s in snapshots_list}

    for snap in snapshots_list:
        status = snap.data_quality
        if snap.warnings:
            status += f" ({', '.join(snap.warnings)})"
        print(f"  {snap.ticker}: {status}")

    print()

    # Preview what the LLM will see
    if args.verbose:
        for snap in snapshots_list:
            print("=" * 60)
            print(snap.to_llm_context())
            print()

    # ------------------------------------------------------------------
    # Step 3: Build slim graph
    # ------------------------------------------------------------------
    print("Initialising SlimTradingAgentsGraph…", flush=True)
    from slim_analysts.graph import SlimTradingAgentsGraph

    graph = SlimTradingAgentsGraph(
        snapshots=snapshots,
        selected_analysts=args.analysts,
    )

    # ------------------------------------------------------------------
    # Step 4: Run reasoning per ticker
    # ------------------------------------------------------------------
    decisions = {}
    for ticker in tickers:
        if ticker not in snapshots or snapshots[ticker].data_quality == "failed":
            print(f"  [{ticker}] Skipped — data quality failed")
            decisions[ticker] = {"action": "SKIP", "conviction": 0.0, "reason": "data_quality_failed"}
            continue

        print(f"  [{ticker}] Running reasoning chain…", flush=True)
        try:
            result = graph.run(ticker, date)
        except Exception as e:
            logger.error(f"[{ticker}] Reasoning chain failed: {e}", exc_info=True)
            print(f"  [{ticker}] ERROR: {e}")
            decisions[ticker] = {"action": "ERROR", "conviction": 0.0, "reason": str(e)}
            continue

        action     = result["action"]
        conviction = result["conviction"]
        print(f"  [{ticker}] → {action}  (conviction {conviction:.2f})")

        if args.verbose:
            print(f"\n  --- Judge Decision ---\n{result['judge_decision']}\n")
            print(f"  --- Final Decision ---\n{result['raw_decision']}\n")

        decisions[ticker] = {
            "ticker":     ticker,
            "date":       date,
            "action":     action,
            "conviction": conviction,
            "raw_decision":   result["raw_decision"],
            "judge_decision": result["judge_decision"],
        }

    # ------------------------------------------------------------------
    # Step 5: Output
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"DECISIONS — {date}")
    print("=" * 50)
    for ticker, d in decisions.items():
        action     = d.get("action", "?")
        conviction = d.get("conviction", 0.0)
        bar_len    = int(conviction * 20)
        bar        = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {ticker:<6}  {action:<4}  [{bar}] {conviction:.0%}")
    print()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(decisions, f, indent=2)
        print(f"Decisions written to {args.output}")


if __name__ == "__main__":
    main()
