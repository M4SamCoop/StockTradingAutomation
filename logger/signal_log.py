"""
Signal logger — SQLite-backed store for trading decisions and their outcomes.

Schema
------
signals   — one row per BUY/SELL/HOLD decision from phase3_runner
outcomes  — one row per (signal, days_forward) pair; filled in by score.py

Typical flow
------------
1. phase3_runner.py calls log_signal() after each ticker's reasoning chain.
2. score.py calls get_unscored_signals() daily, fetches forward prices via
   yfinance, and calls write_outcome() for 1d / 3d / 5d horizons.
3. score.py aggregates outcomes into a hit-rate / avg-return table.
"""

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default DB location — sits alongside the project, not in the repo
_DEFAULT_DB = Path.home() / ".tradingagents" / "signal_log.db"

# Horizon days we track outcomes for
OUTCOME_HORIZONS = (1, 3, 5)


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at           TEXT NOT NULL,          -- ISO-8601 UTC
    ticker              TEXT NOT NULL,
    signal_date         TEXT NOT NULL,          -- YYYY-MM-DD analysis date
    action              TEXT NOT NULL,          -- BUY | SELL | HOLD
    conviction          REAL NOT NULL,          -- 0.0 – 1.0
    entry_price         REAL,                   -- close price at signal time
    judge_decision      TEXT,
    raw_decision        TEXT,
    market_report       TEXT,
    fundamentals_report TEXT,
    sentiment_report    TEXT,
    news_report         TEXT
);

CREATE TABLE IF NOT EXISTS outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id       INTEGER NOT NULL REFERENCES signals(id),
    evaluated_at    TEXT NOT NULL,              -- ISO-8601 UTC when scored
    days_forward    INTEGER NOT NULL,           -- 1 | 3 | 5
    price_then      REAL,                       -- close price N days after signal
    pct_change      REAL,                       -- (price_then - entry_price) / entry_price
    correct         INTEGER,                    -- 1 correct / 0 wrong / NULL for HOLD
    UNIQUE(signal_id, days_forward)
);

CREATE INDEX IF NOT EXISTS idx_signals_ticker      ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_signal_date ON signals(signal_date);
CREATE INDEX IF NOT EXISTS idx_signals_action      ON signals(action);
"""


def get_db_path(db_path: Optional[Path] = None) -> Path:
    path = Path(db_path) if db_path else _DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _connect(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> Path:
    """Create tables if they don't exist. Returns the resolved DB path."""
    path = get_db_path(db_path)
    with _connect(path) as conn:
        conn.executescript(_SCHEMA)
    return path


# ---------------------------------------------------------------------------
# Write API
# ---------------------------------------------------------------------------

def log_signal(
    ticker: str,
    signal_date: str,
    action: str,
    conviction: float,
    entry_price: Optional[float] = None,
    judge_decision: str = "",
    raw_decision: str = "",
    market_report: str = "",
    fundamentals_report: str = "",
    sentiment_report: str = "",
    news_report: str = "",
    db_path: Optional[Path] = None,
) -> int:
    """
    Persist one trading decision.

    Returns the new row's signal id.
    """
    path = init_db(db_path)
    logged_at = datetime.now(timezone.utc).isoformat()

    with _connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO signals
                (logged_at, ticker, signal_date, action, conviction, entry_price,
                 judge_decision, raw_decision, market_report, fundamentals_report,
                 sentiment_report, news_report)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                logged_at, ticker.upper(), signal_date, action.upper(),
                round(float(conviction), 4), entry_price,
                judge_decision, raw_decision,
                market_report, fundamentals_report, sentiment_report, news_report,
            ),
        )
        signal_id = cur.lastrowid

    logger.info(f"Logged signal #{signal_id}: {ticker} {action} {conviction:.0%} on {signal_date}")
    return signal_id


def write_outcome(
    signal_id: int,
    days_forward: int,
    price_then: Optional[float],
    pct_change: Optional[float],
    correct: Optional[int],
    db_path: Optional[Path] = None,
) -> None:
    """
    Upsert an outcome row for a given signal + horizon.
    Silently overwrites if the (signal_id, days_forward) pair already exists.
    """
    path = get_db_path(db_path)
    evaluated_at = datetime.now(timezone.utc).isoformat()

    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO outcomes
                (signal_id, evaluated_at, days_forward, price_then, pct_change, correct)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(signal_id, days_forward) DO UPDATE SET
                evaluated_at = excluded.evaluated_at,
                price_then   = excluded.price_then,
                pct_change   = excluded.pct_change,
                correct      = excluded.correct
            """,
            (signal_id, evaluated_at, days_forward, price_then, pct_change, correct),
        )


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------

def get_unscored_signals(
    min_days_old: int = 1,
    db_path: Optional[Path] = None,
) -> list[dict]:
    """
    Return signals that are missing at least one outcome horizon and are old
    enough for that horizon to have elapsed.

    Each returned dict has all signal columns plus a 'missing_horizons' list.
    """
    path = get_db_path(db_path)
    if not path.exists():
        return []

    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT s.*
            FROM signals s
            WHERE date(s.signal_date) <= date('now', ? || ' days')
            ORDER BY s.signal_date ASC
            """,
            (f"-{min_days_old}",),
        ).fetchall()

        results = []
        for row in rows:
            sid = row["id"]
            scored = {
                r["days_forward"]
                for r in conn.execute(
                    "SELECT days_forward FROM outcomes WHERE signal_id = ?", (sid,)
                ).fetchall()
            }
            missing = [h for h in OUTCOME_HORIZONS if h not in scored]
            if missing:
                d = dict(row)
                d["missing_horizons"] = missing
                results.append(d)

    return results


def get_all_signals(db_path: Optional[Path] = None) -> list[dict]:
    """Return every signal row, newest first."""
    path = get_db_path(db_path)
    if not path.exists():
        return []
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY signal_date DESC, logged_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_outcomes(db_path: Optional[Path] = None) -> list[dict]:
    """Return all outcome rows joined with their signal metadata."""
    path = get_db_path(db_path)
    if not path.exists():
        return []
    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT
                s.id as signal_id, s.ticker, s.signal_date, s.action,
                s.conviction, s.entry_price,
                o.days_forward, o.price_then, o.pct_change, o.correct,
                o.evaluated_at
            FROM outcomes o
            JOIN signals s ON s.id = o.signal_id
            ORDER BY s.signal_date DESC, o.days_forward ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]
