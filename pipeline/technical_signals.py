"""
Technical signal computation — replaces TradingAgents' Technical Analyst agent.

Uses the `ta` library (pip install ta) to compute the same indicator set the
Technical Analyst was producing via LLM tool calls, but locally in milliseconds
for free. Compatible with Python 3.10+.

Returns a TechnicalSignals dataclass rather than a verbose prose report.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import ta

from pipeline.models import TechnicalSignals

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_technical_signals(
    ticker: str,
    date: Optional[str] = None,
    lookback_days: int = 300,
) -> Optional[TechnicalSignals]:
    """
    Compute technical indicators for a ticker as of a given date.

    Args:
        ticker:        Ticker symbol, e.g. "AAPL"
        date:          Analysis date as "YYYY-MM-DD". Defaults to today.
        lookback_days: How many calendar days of history to fetch.
                       300 days is enough for 200-day SMA with buffer.

    Returns:
        TechnicalSignals dataclass, or None if data fetch fails.
    """
    date = date or datetime.today().strftime("%Y-%m-%d")

    df = _fetch_ohlcv(ticker, date, lookback_days)
    if df is None or df.empty:
        logger.warning(f"[{ticker}] No OHLCV data returned.")
        return None

    try:
        return _compute_signals(ticker, date, df)
    except Exception as e:
        logger.error(f"[{ticker}] Error computing signals: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def _fetch_ohlcv(ticker: str, end_date: str, lookback_days: int) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data via yfinance.

    Uses yf.Ticker().history() rather than yf.download() — the Ticker instance
    is per-object so it's thread-safe when called from a ThreadPoolExecutor.
    yf.download() uses global state that causes data cross-contamination under
    concurrent calls (different tickers receiving the same result).
    """
    try:
        end = datetime.strptime(end_date, "%Y-%m-%d")
        start = end - timedelta(days=lookback_days)

        raw = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
        )

        if raw.empty:
            return None

        raw = raw.rename(columns=str.lower)
        # history() returns: Open High Low Close Volume Dividends Stock Splits
        raw = raw[["open", "high", "low", "close", "volume"]].dropna()
        raw.index = pd.to_datetime(raw.index)
        # Strip timezone info — keeps downstream date handling simple
        raw.index = raw.index.tz_localize(None)
        return raw

    except Exception as e:
        logger.error(f"[{ticker}] OHLCV fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _compute_signals(ticker: str, date: str, df: pd.DataFrame) -> TechnicalSignals:
    """Compute all indicators and extract the latest values."""

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # --- Moving averages ---
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema10  = close.ewm(span=10, adjust=False).mean()

    # --- RSI ---
    rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # --- MACD ---
    macd_ind = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd_line   = macd_ind.macd()
    macd_signal = macd_ind.macd_signal()
    macd_hist   = macd_ind.macd_diff()

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_upper  = bb.bollinger_hband()
    bb_mid    = bb.bollinger_mavg()
    bb_lower  = bb.bollinger_lband()

    # --- ATR ---
    atr_series = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # --- VWMA (manual — volume-weighted moving average) ---
    vwma_20 = (close * vol).rolling(20).sum() / vol.rolling(20).sum()

    # --- Volume SMA ---
    vol_sma_20 = vol.rolling(20).mean()

    # --- Extract last row values ---
    def last(series) -> Optional[float]:
        if series is None or series.empty:
            return None
        val = series.iloc[-1]
        try:
            f = float(val)
            return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
        except (TypeError, ValueError):
            return None

    c       = last(close)
    s50     = last(sma50)
    s200    = last(sma200)
    e10     = last(ema10)
    rsi_val = last(rsi_series)
    macd_v  = last(macd_line)
    macd_s  = last(macd_signal)
    macd_h  = last(macd_hist)
    bbu     = last(bb_upper)
    bbm     = last(bb_mid)
    bbl     = last(bb_lower)
    atr_v   = last(atr_series)
    vwma_v  = last(vwma_20)
    vol_v   = last(vol)
    vol_sma = last(vol_sma_20)

    # --- Derived ---
    price_vs_sma50  = _pct(c, s50)
    price_vs_sma200 = _pct(c, s200)
    atr_pct         = round(atr_v / c * 100, 4) if (atr_v and c) else None
    volume_ratio    = round(vol_v / vol_sma, 4) if (vol_v and vol_sma and vol_sma > 0) else None

    bb_pct = None
    if all(v is not None for v in [c, bbu, bbl]):
        band_width = bbu - bbl
        if band_width > 0:
            bb_pct = round((c - bbl) / band_width, 4)

    # MACD crossover: compare last two rows
    macd_crossover = "none"
    if len(df) >= 2:
        prev_macd_v = macd_line.iloc[-2]
        prev_macd_s = macd_signal.iloc[-2]
        if all(v is not None and not (isinstance(v, float) and np.isnan(v))
               for v in [macd_v, macd_s, prev_macd_v, prev_macd_s]):
            if float(prev_macd_v) < float(prev_macd_s) and macd_v > macd_s:
                macd_crossover = "bullish"
            elif float(prev_macd_v) > float(prev_macd_s) and macd_v < macd_s:
                macd_crossover = "bearish"

    trend    = _classify_trend(c, s50, s200)
    momentum = _classify_momentum(rsi_val, macd_h)

    last_date = str(df.index[-1].date())

    return TechnicalSignals(
        ticker=ticker,
        date=last_date,
        close=c,
        open=last(df["open"]),
        high=last(high),
        low=last(low),
        volume=vol_v,
        sma_50=s50,
        sma_200=s200,
        ema_10=e10,
        price_vs_sma50_pct=price_vs_sma50,
        price_vs_sma200_pct=price_vs_sma200,
        rsi_14=rsi_val,
        macd=macd_v,
        macd_signal=macd_s,
        macd_hist=macd_h,
        macd_crossover=macd_crossover,
        bb_upper=bbu,
        bb_mid=bbm,
        bb_lower=bbl,
        bb_pct=bb_pct,
        atr_14=atr_v,
        atr_pct=atr_pct,
        vwma_20=vwma_v,
        volume_sma_20=vol_sma,
        volume_ratio=volume_ratio,
        trend=trend,
        momentum=momentum,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(value: Optional[float], reference: Optional[float]) -> Optional[float]:
    if value is None or reference is None or reference == 0:
        return None
    return round((value - reference) / reference * 100, 2)


def _classify_trend(
    close: Optional[float],
    sma50: Optional[float],
    sma200: Optional[float],
) -> Optional[str]:
    if close is None:
        return None
    above50  = (close > sma50)  if sma50  is not None else None
    above200 = (close > sma200) if sma200 is not None else None

    if above50 is True and above200 is True:
        return "uptrend"
    if above50 is False and above200 is False:
        return "downtrend"
    if above50 is not None or above200 is not None:
        return "mixed"
    return "sideways"


def _classify_momentum(
    rsi: Optional[float],
    macd_hist: Optional[float],
) -> Optional[str]:
    if rsi is None and macd_hist is None:
        return None
    bull, bear = 0, 0
    if rsi is not None:
        if rsi > 55:
            bull += 1
        elif rsi < 45:
            bear += 1
    if macd_hist is not None:
        if macd_hist > 0:
            bull += 1
        elif macd_hist < 0:
            bear += 1
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    return "neutral"
