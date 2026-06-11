"""
Formatters — convert TickerSnapshot fields into concise analyst report strings.

These are pure functions (no LLM, no I/O). They produce the report text that
slim analyst nodes return directly, replacing the verbose prose that TradingAgents'
data-gathering agents would otherwise generate via multiple LLM + tool calls.

Format is readable prose / structured text (not raw JSON) so the bull/bear
researchers can reason over it naturally.
"""

from typing import Optional
from pipeline.models import TickerSnapshot


# ---------------------------------------------------------------------------
# Public formatters — one per analyst type
# ---------------------------------------------------------------------------

def format_technical_report(snap: TickerSnapshot) -> str:
    t = snap.technical
    if t is None:
        return f"TECHNICAL SUMMARY — {snap.ticker} — {snap.analysis_date}\n\nNo technical data available."

    lines = [f"TECHNICAL SUMMARY — {snap.ticker} — {t.date}"]
    lines.append("")

    # Price action
    trend_str = (t.trend or "unknown").upper()
    vs50  = f"{t.price_vs_sma50_pct:+.1f}% vs 50d SMA"  if t.price_vs_sma50_pct  is not None else ""
    vs200 = f"{t.price_vs_sma200_pct:+.1f}% vs 200d SMA" if t.price_vs_sma200_pct is not None else ""
    trend_detail = " | ".join(filter(None, [vs50, vs200]))
    lines.append(f"Price Action: ${t.close}  |  Trend: {trend_str} ({trend_detail})")

    # Momentum
    rsi_str = f"RSI(14): {t.rsi_14:.1f}" if t.rsi_14 is not None else "RSI: N/A"
    rsi_label = ""
    if t.rsi_14 is not None:
        if t.rsi_14 > 70:   rsi_label = " — overbought"
        elif t.rsi_14 < 30: rsi_label = " — oversold"
        elif t.rsi_14 > 55: rsi_label = " — bullish momentum"
        elif t.rsi_14 < 45: rsi_label = " — weakening momentum"
        else:               rsi_label = " — neutral"
    lines.append(f"{rsi_str}{rsi_label}")

    if t.macd is not None:
        cross_str = f"  [{t.macd_crossover.upper()} CROSSOVER]" if t.macd_crossover not in (None, "none") else ""
        lines.append(f"MACD: {t.macd:.4f} / Signal: {t.macd_signal:.4f} / Hist: {t.macd_hist:.4f}{cross_str}")

    # Volatility
    bb_line = []
    if t.bb_pct is not None:
        bb_line.append(f"BB%: {t.bb_pct:.2f} ({'near lower band' if t.bb_pct < 0.2 else 'near upper band' if t.bb_pct > 0.8 else 'mid-band'})")
    if t.atr_14 is not None:
        bb_line.append(f"ATR(14): ${t.atr_14:.2f} ({t.atr_pct:.2f}% of price)" if t.atr_pct else f"ATR(14): ${t.atr_14:.2f}")
    if bb_line:
        lines.append(" | ".join(bb_line))

    # Volume
    if t.volume_ratio is not None:
        vol_label = "above average" if t.volume_ratio > 1.3 else "below average"
        lines.append(f"Volume: {t.volume_ratio:.2f}× 20d average — {vol_label}")
    if t.vwma_20 is not None:
        vwma_rel = "below VWMA" if t.close < t.vwma_20 else "above VWMA"
        lines.append(f"VWMA(20): ${t.vwma_20:.2f} — price {vwma_rel} (${t.close - t.vwma_20:+.2f})")

    # Key levels
    levels = []
    support_candidates = [
        (t.bb_lower, "BB Lower"),
        (t.sma_50,   "50d SMA"),
        (t.sma_200,  "200d SMA"),
    ]
    resist_candidates = [
        (t.ema_10,   "EMA10"),
        (t.vwma_20,  "VWMA20"),
        (t.bb_upper, "BB Upper"),
    ]
    support = " / ".join(
        f"${v:.2f} ({lbl})" for v, lbl in support_candidates
        if v is not None and t.close is not None and v <= t.close
    ) or "N/A"
    resist = " / ".join(
        f"${v:.2f} ({lbl})" for v, lbl in resist_candidates
        if v is not None and t.close is not None and v >= t.close
    ) or "N/A"
    lines.append(f"\nKey Levels:")
    lines.append(f"  Support:    {support}")
    lines.append(f"  Resistance: {resist}")

    lines.append(f"\nMomentum Summary: {(t.momentum or 'neutral').upper()}")

    return "\n".join(lines)


def format_fundamentals_report(snap: TickerSnapshot) -> str:
    f = snap.fundamentals
    if f is None:
        return f"FUNDAMENTALS SUMMARY — {snap.ticker} — {snap.analysis_date}\n\nNo fundamental data available."

    lines = [f"FUNDAMENTALS SUMMARY — {snap.ticker} — {snap.analysis_date}"]
    lines.append("")

    # Valuation
    val_parts = []
    if f.pe_ratio:        val_parts.append(f"PE {f.pe_ratio:.1f}")
    if f.forward_pe:      val_parts.append(f"Fwd PE {f.forward_pe:.1f}")
    if f.pb_ratio:        val_parts.append(f"P/B {f.pb_ratio:.2f}")
    if f.ps_ratio:        val_parts.append(f"P/S {f.ps_ratio:.2f}")
    if f.ev_ebitda:       val_parts.append(f"EV/EBITDA {f.ev_ebitda:.1f}")
    lines.append("Valuation:    " + " | ".join(val_parts) if val_parts else "Valuation: N/A")

    # Profitability
    prof_parts = []
    if f.profit_margin:   prof_parts.append(f"Net Margin {f.profit_margin*100:.1f}%")
    if f.operating_margin:prof_parts.append(f"Op Margin {f.operating_margin*100:.1f}%")
    if f.roe:             prof_parts.append(f"ROE {f.roe*100:.1f}%")
    if f.roa:             prof_parts.append(f"ROA {f.roa*100:.1f}%")
    lines.append("Profitability: " + " | ".join(prof_parts) if prof_parts else "Profitability: N/A")

    # Growth
    growth_parts = []
    if f.revenue_growth_yoy is not None:
        growth_parts.append(f"Revenue {f.revenue_growth_yoy*100:+.1f}% YoY")
    if f.earnings_growth_yoy is not None:
        growth_parts.append(f"Earnings {f.earnings_growth_yoy*100:+.1f}% YoY")
    lines.append("Growth:        " + " | ".join(growth_parts) if growth_parts else "Growth: N/A")

    # Balance sheet
    bs_parts = []
    if f.debt_to_equity is not None: bs_parts.append(f"D/E {f.debt_to_equity:.1f}")
    if f.current_ratio is not None:  bs_parts.append(f"Current Ratio {f.current_ratio:.2f}")
    if f.quick_ratio is not None:    bs_parts.append(f"Quick Ratio {f.quick_ratio:.2f}")
    lines.append("Balance Sheet: " + " | ".join(bs_parts) if bs_parts else "Balance Sheet: N/A")

    # EPS
    eps_parts = []
    if f.eps_ttm:     eps_parts.append(f"TTM ${f.eps_ttm:.2f}")
    if f.eps_forward: eps_parts.append(f"Fwd ${f.eps_forward:.2f}")
    if f.next_earnings_date: eps_parts.append(f"Next Earnings: {f.next_earnings_date}")
    lines.append("EPS:           " + " | ".join(eps_parts) if eps_parts else "EPS: N/A")

    # Cap + sector
    if f.market_cap:
        cap_str = f"${f.market_cap/1e12:.2f}T" if f.market_cap >= 1e12 else f"${f.market_cap/1e9:.1f}B"
        lines.append(f"Market Cap:    {cap_str}")
    if f.sector:
        lines.append(f"Sector:        {f.sector} / {f.industry or 'N/A'}")

    # Quality
    lines.append(f"\nQuality Score: {f.quality_score}/100")
    if f.quality_flags:
        lines.append(f"Flags: {', '.join(f.quality_flags)}")

    return "\n".join(lines)


def format_sentiment_report(snap: TickerSnapshot) -> str:
    s = snap.sentiment
    lines = [f"SENTIMENT & SOCIAL SUMMARY — {snap.ticker} — {snap.analysis_date}"]
    lines.append("")

    if s is None or s.score is None:
        if s and s.headline_count == 0:
            lines.append("Headlines: none available in lookback window")
        else:
            lines.append("Sentiment scoring unavailable (FinBERT not installed)")
        if s and s.news_catalyst and s.news_catalyst != "none":
            lines.append(f"Catalyst detected: {s.news_catalyst.upper()}")
            if s.catalyst_detail:
                lines.append(f"Detail: {s.catalyst_detail}")
        else:
            lines.append("No catalyst detected.")
        return "\n".join(lines)

    label = (s.label or "neutral").upper()
    lines.append(f"Sentiment: {label} (score {s.score:+.3f})")
    lines.append(f"Breakdown: Positive {s.positive_pct:.1f}% | Negative {s.negative_pct:.1f}% | Neutral {s.neutral_pct:.1f}%")
    lines.append(f"Headlines analyzed: {s.headline_count}")

    if s.top_headlines:
        lines.append("\nMost opinionated headlines:")
        for h in s.top_headlines:
            lines.append(f"  • {h}")

    catalyst = s.news_catalyst or "none"
    lines.append(f"\nCatalyst: {catalyst.upper()}")
    if s.catalyst_detail:
        lines.append(f"Detail: {s.catalyst_detail}")

    return "\n".join(lines)


def format_news_report(snap: TickerSnapshot) -> str:
    """
    News report is derived from the same sentiment/news data.
    Focuses on catalyst and headline content rather than sentiment scores.
    """
    s = snap.sentiment
    lines = [f"NEWS SUMMARY — {snap.ticker} — {snap.analysis_date}"]
    lines.append("")

    if s is None or (not s.top_headlines and s.headline_count == 0):
        lines.append("No recent news available in the lookback window.")
        return "\n".join(lines)

    catalyst = (s.news_catalyst or "none")
    if catalyst != "none":
        lines.append(f"PRIMARY CATALYST: {catalyst.upper()}")
        if s.catalyst_detail:
            lines.append(f"  {s.catalyst_detail}")
        lines.append("")

    if s.top_headlines:
        lines.append("Recent headlines:")
        for h in s.top_headlines:
            lines.append(f"  • {h}")
    elif s.headline_count and s.headline_count > 0:
        lines.append(f"({s.headline_count} headlines analyzed — see sentiment report for details)")
    else:
        lines.append("No headlines available.")

    return "\n".join(lines)
