"""
Score logged signals against subsequent price action.

Usage
-----
    # Fetch outcomes for any signals old enough to be scored
    python3 -m logger.score

    # Show summary stats only (no new fetches)
    python3 -m logger.score --report

    # Use a non-default DB
    python3 -m logger.score --db ~/my_signals.db

What it does
------------
1. Calls get_unscored_signals() to find signals missing outcome rows.
2. For each missing horizon (1d / 3d / 5d), fetches the closing price that
   many trading days after signal_date via yfinance.
3. Writes outcome rows via write_outcome().
4. Prints a formatted report: per-horizon hit rate, avg return, and a
   conviction-bucketed breakdown.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf

from logger.signal_log import (
    OUTCOME_HORIZONS,
    get_db_path,
    get_outcomes,
    get_unscored_signals,
    write_outcome,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

def _fetch_close_n_trading_days_after(
    ticker: str,
    signal_date: str,
    n_days: int,
    max_calendar_days: int = 14,
) -> Optional[float]:
    """
    Return the closing price N *trading* days after signal_date.

    We fetch a window of max_calendar_days calendar days to handle weekends
    and holidays, then index into the Nth row.

    Returns None if data is unavailable (e.g. future date, delisted ticker).
    """
    try:
        start = datetime.strptime(signal_date, "%Y-%m-%d") + timedelta(days=1)
        end   = start + timedelta(days=max_calendar_days)

        df = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )

        if df is None or df.empty:
            return None

        df = df.rename(columns=str.lower)
        if len(df) < n_days:
            # Not enough trading days in window — data not available yet
            return None

        return float(round(df["close"].iloc[n_days - 1], 4))

    except Exception as e:
        logger.warning(f"[{ticker}] Price fetch failed for +{n_days}d: {e}")
        return None


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def _is_correct(action: str, pct_change: Optional[float]) -> Optional[int]:
    """
    1 if the signal direction matches actual price movement.
    0 if it didn't.
    None for HOLD (direction not predictive).
    """
    if pct_change is None:
        return None
    if action == "BUY":
        return 1 if pct_change > 0 else 0
    if action == "SELL":
        return 1 if pct_change < 0 else 0
    return None  # HOLD


def fetch_and_store_outcomes(db_path: Optional[Path] = None) -> int:
    """
    Score all unscored signals. Returns the number of outcome rows written.
    """
    unscored = get_unscored_signals(min_days_old=1, db_path=db_path)

    if not unscored:
        print("No signals pending scoring.")
        return 0

    written = 0
    for sig in unscored:
        ticker      = sig["ticker"]
        signal_date = sig["signal_date"]
        action      = sig["action"]
        entry_price = sig["entry_price"]
        sid         = sig["id"]

        for horizon in sig["missing_horizons"]:
            print(f"  Scoring {ticker} signal #{sid} ({action}, {signal_date}) +{horizon}d…", end=" ", flush=True)

            price_then = _fetch_close_n_trading_days_after(ticker, signal_date, horizon)

            if price_then is None:
                print("no data yet")
                continue

            pct_change = (
                round((price_then - entry_price) / entry_price, 6)
                if entry_price and entry_price > 0
                else None
            )
            correct = _is_correct(action, pct_change)

            write_outcome(
                signal_id=sid,
                days_forward=horizon,
                price_then=price_then,
                pct_change=pct_change,
                correct=correct,
                db_path=db_path,
            )
            written += 1

            direction = "✓" if correct == 1 else ("✗" if correct == 0 else "—")
            pct_str = f"{pct_change*100:+.2f}%" if pct_change is not None else "N/A"
            print(f"${price_then:.2f}  {pct_str}  {direction}")

    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(db_path: Optional[Path] = None) -> None:
    """Print a summary of all scored outcomes."""
    rows = get_outcomes(db_path=db_path)

    if not rows:
        print("No scored outcomes yet. Run without --report to fetch prices first.")
        return

    # ---- Per-horizon stats ------------------------------------------------
    print("\n" + "=" * 60)
    print("SIGNAL PERFORMANCE REPORT")
    print("=" * 60)

    for horizon in OUTCOME_HORIZONS:
        h_rows = [r for r in rows if r["days_forward"] == horizon]
        directional = [r for r in h_rows if r["correct"] is not None]  # exclude HOLD

        if not h_rows:
            continue

        n_total      = len(h_rows)
        n_dir        = len(directional)
        hit_rate     = sum(r["correct"] for r in directional) / n_dir if n_dir else None
        avg_return   = (
            sum(r["pct_change"] for r in h_rows if r["pct_change"] is not None)
            / sum(1 for r in h_rows if r["pct_change"] is not None)
            if any(r["pct_change"] is not None for r in h_rows)
            else None
        )

        print(f"\n+{horizon}d Horizon  ({n_total} outcomes, {n_dir} directional)")
        print(f"  Hit rate:   {hit_rate*100:.1f}%  ({sum(r['correct'] for r in directional)}/{n_dir})" if hit_rate is not None else "  Hit rate:   N/A")
        print(f"  Avg return: {avg_return*100:+.2f}%" if avg_return is not None else "  Avg return: N/A")

        # Per-action breakdown
        for action in ("BUY", "SELL", "HOLD"):
            a_rows = [r for r in h_rows if r["action"] == action]
            if not a_rows:
                continue
            a_dir  = [r for r in a_rows if r["correct"] is not None]
            a_hit  = sum(r["correct"] for r in a_dir) / len(a_dir) if a_dir else None
            a_ret  = (
                sum(r["pct_change"] for r in a_rows if r["pct_change"] is not None)
                / sum(1 for r in a_rows if r["pct_change"] is not None)
                if any(r["pct_change"] is not None for r in a_rows)
                else None
            )
            hit_str = f"{a_hit*100:.0f}%" if a_hit is not None else " — "
            ret_str = f"{a_ret*100:+.2f}%" if a_ret is not None else "  N/A"
            print(f"    {action:<4}  n={len(a_rows):<3}  hit={hit_str:<6}  avg={ret_str}")

    # ---- Conviction buckets (1d only) ------------------------------------
    h1 = [r for r in rows if r["days_forward"] == 1 and r["pct_change"] is not None]
    if len(h1) >= 5:
        print("\nConviction buckets (+1d):")
        buckets = [
            ("High   (≥0.75)", lambda r: r["conviction"] >= 0.75),
            ("Medium (0.5–0.75)", lambda r: 0.5 <= r["conviction"] < 0.75),
            ("Low    (<0.5)",  lambda r: r["conviction"] < 0.5),
        ]
        for label, fn in buckets:
            b = [r for r in h1 if fn(r)]
            if not b:
                continue
            b_dir = [r for r in b if r["correct"] is not None]
            b_hit = sum(r["correct"] for r in b_dir) / len(b_dir) if b_dir else None
            b_ret = sum(r["pct_change"] for r in b) / len(b)
            hit_str = f"{b_hit*100:.0f}%" if b_hit is not None else "—"
            print(f"  {label}  n={len(b):<3}  hit={hit_str:<6}  avg={b_ret*100:+.2f}%")

    # ---- Recent signals table --------------------------------------------
    print("\nRecent signals (last 10):")
    print(f"  {'Date':<12} {'Ticker':<6} {'Action':<5} {'Conv':>5}  {'1d':>6}  {'3d':>6}  {'5d':>6}")
    print("  " + "-" * 52)

    # Group outcomes by signal
    by_signal: dict[int, dict] = {}
    for r in rows:
        sid = r["signal_id"]
        if sid not in by_signal:
            by_signal[sid] = {
                "ticker": r["ticker"], "date": r["signal_date"],
                "action": r["action"], "conviction": r["conviction"],
                "horizons": {},
            }
        by_signal[sid]["horizons"][r["days_forward"]] = r["pct_change"]

    recent = sorted(by_signal.values(), key=lambda x: x["date"], reverse=True)[:10]
    for s in recent:
        h = s["horizons"]
        def fmt(v): return f"{v*100:+.1f}%" if v is not None else "  —   "
        print(
            f"  {s['date']:<12} {s['ticker']:<6} {s['action']:<5} "
            f"{s['conviction']:>4.0%}  "
            f"{fmt(h.get(1)):>6}  {fmt(h.get(3)):>6}  {fmt(h.get(5)):>6}"
        )

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Score logged signals against price outcomes.")
    p.add_argument("--report", action="store_true", help="Print report only; skip fetching new outcomes.")
    p.add_argument("--db", default=None, metavar="FILE", help="Path to SQLite DB (default: ~/.tradingagents/signal_log.db).")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    args = _parse_args()
    db_path = Path(args.db) if args.db else None

    if not args.report:
        print("Fetching outcomes for unscored signals…")
        written = fetch_and_store_outcomes(db_path=db_path)
        if written:
            print(f"\nWrote {written} new outcome row(s).")

    print_report(db_path=db_path)


if __name__ == "__main__":
    main()
