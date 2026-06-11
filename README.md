# StockTradingAutomation

An AI-powered stock analysis and trading system built on a **hybrid pipeline** that combines free local computation with [TradingAgents](https://github.com/TauricResearch/TradingAgents) LLM reasoning — cutting API costs by ~90% vs running TradingAgents stock.

The key idea: replace TradingAgents' data-gathering analyst agents (which burn tokens calling APIs and computing indicators) with local pre-computation (pandas `ta`, yfinance, FinBERT). Only the reasoning/debate layer runs on LLMs.

---

## Architecture

```
Phase 1: Screener + Data Assembly
  ├── Screener  — scores 100+ tickers via local technical + fundamental heuristics
  ├── Technical — SMA50/200, EMA10, RSI, MACD, Bollinger Bands, ATR, VWMA (ta library)
  └── Fundamentals — PE, margins, growth, balance sheet (yfinance)

Phase 2: Sentiment (FinBERT)
  ├── Headlines — fetched via yfinance.Ticker.news
  ├── Sentiment scoring — ProsusAI/finbert runs locally (CPU, ~420MB one-time download)
  └── Catalyst detection — rule-based regex: earnings, FDA, upgrade/downgrade, insider, M&A…

Phase 3: LLM Reasoning (TradingAgents slim mode)
  ├── Slim analyst nodes inject pre-computed data instead of calling tools
  ├── Bull/Bear debate  — claude-haiku-4-5  (quick_think_llm)
  ├── Research Manager  — claude-sonnet-4-6 (deep_think_llm)
  ├── Trader + Risk     — claude-haiku-4-5
  └── Portfolio Manager — claude-sonnet-4-6

Phase 4: Execution (planned)
  └── Alpaca brokerage — paper/live trading via alpaca-py
```

**Cost comparison** (per ticker per run):

| Approach | Tokens | Est. cost |
|---|---|---|
| Stock TradingAgents | ~50,000 input tokens | ~$0.15–0.40 |
| This hybrid pipeline | ~1,500 input tokens | ~$0.005–0.015 |

---

## Quickstart

### Prerequisites

- Python **3.12+** (required by TradingAgents)
- Anthropic API key

### Install

```bash
git clone https://github.com/M4SamCoop/StockTradingAutomation.git
cd StockTradingAutomation

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Environment

```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
# ALPACA_API_KEY=...       # Phase 4 — uncomment when ready
# ALPACA_SECRET_KEY=...    # Phase 4
```

---

## Usage

### Phase 1 + 2: Data pipeline only (no LLM cost)

```bash
# Run on specific tickers
python3 run_pipeline.py --tickers AAPL MSFT NVDA

# Run screener on S&P 500 high-liquidity watchlist, pick top 5
python3 run_pipeline.py --screen --shortlist-size 5

# Skip fundamentals for speed
python3 run_pipeline.py --tickers AAPL --no-fundamentals

# Write output to JSON
python3 run_pipeline.py --tickers AAPL --output output/aapl.json
```

### Phase 3: Full reasoning chain

```bash
# Run TradingAgents reasoning on pre-assembled snapshots
python3 phase3_runner.py --tickers AAPL MSFT --date 2026-06-11

# With screener
python3 phase3_runner.py --screen --shortlist-size 3 --date 2026-06-11

# Verbose (prints full reports, judge decisions, debate output)
python3 phase3_runner.py --tickers AAPL --verbose

# Save decisions to JSON
python3 phase3_runner.py --tickers AAPL MSFT --output decisions/run.json
```

Output:
```
DECISIONS — 2026-06-11
==================================================
  AAPL    BUY   [████████████████░░░░] 80%
  MSFT    HOLD  [████████████░░░░░░░░] 60%
```

---

## Project Structure

```
.
├── config/
│   └── watchlist.py          # SP500_HIGH_LIQUIDITY — 103 tickers
├── pipeline/
│   ├── models.py             # TickerSnapshot, TechnicalSignals, etc.
│   ├── screener.py           # Multi-factor screener (parallel)
│   ├── technical_signals.py  # ta library: RSI, MACD, BB, ATR, VWMA
│   ├── fundamentals_pull.py  # yfinance: PE, margins, growth, quality score
│   ├── news_fetch.py         # yfinance news + catalyst regex detection
│   ├── sentiment_engine.py   # FinBERT local inference
│   └── data_assembler.py     # Orchestrates all of the above
├── slim_analysts/
│   ├── formatters.py         # TickerSnapshot → compact report strings
│   ├── nodes.py              # Slim analyst node factories (no LLM tool calls)
│   └── graph.py              # SlimTradingAgentsGraph + monkey-patch context mgr
├── run_pipeline.py           # CLI: Phases 1+2 (data only)
└── phase3_runner.py          # CLI: Full pipeline through Phase 3 reasoning
```

---

## How the Slim Analyst Trick Works

TradingAgents' four analyst agents each loop: `call tool → process result → call tool → …` until they've gathered enough data, then write a report. Each loop burns 3–10 LLM calls.

We replace those agents at the graph level using a **monkey-patch context manager**:

```python
# tradingagents/graph/setup.py uses `from tradingagents.agents import *`
# Those names live in setup.py's module namespace — patching them there
# means setup_graph() picks up our slim factories automatically.

import tradingagents.graph.setup as _ta_setup

with _patch_analysts(snapshots):       # swap in slim factories
    ta_graph = TradingAgentsGraph(...) # __init__ calls setup_graph()
                                       # slim factories get baked into the graph
```

Each slim factory returns a node that immediately writes a pre-formatted report to the AgentState and returns an `AIMessage` with **no `tool_calls`**. TradingAgents' conditional logic sees no tool calls → routes directly to `Msg Clear` → moves on to the researcher debate. Zero tool call loops.

---

## Roadmap

- [x] Phase 1: Screener + technical/fundamental data assembly
- [x] Phase 2: FinBERT sentiment + catalyst detection
- [x] Phase 3: Slim TradingAgents integration
- [ ] Phase 4: Alpaca brokerage execution
- [ ] Phase 5: Scheduling (pre-market daily scan)
- [ ] Phase 6: Portfolio tracking + risk controls
- [ ] Phase 7: Backtesting on historical dates

---

## Tech Stack

| Layer | Tool |
|---|---|
| Reasoning engine | [TradingAgents](https://github.com/TauricResearch/TradingAgents) (slim mode) |
| LLM | Claude Haiku 4.5 (analysis) + Sonnet 4.6 (judgment) |
| Technical indicators | `ta` library (pandas-based, Python 3.12 compatible) |
| Sentiment | `ProsusAI/finbert` via HuggingFace Transformers |
| Market data | yfinance (thread-safe: `Ticker().history()`) |
| Brokerage (planned) | Alpaca via `alpaca-py` |
| Language | Python 3.12+ |

---

## Dependency

This project depends on **TradingAgents** by Tauric Research:
https://github.com/TauricResearch/TradingAgents

---

## License

MIT
