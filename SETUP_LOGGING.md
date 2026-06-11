# Automated Signal Logging — Setup Guide
*Target: Manjaro Linux (KDE). Optimised for execution by Claude Code.*

This sets up the pipeline to run automatically every weekday pre-market,
log BUY/SELL/HOLD decisions to SQLite, and let you score outcomes after the fact.
No paper trading yet — that comes after signal quality is validated.

---

## What gets set up

```
Daily cron job (weekdays, pre-market)
    └── phase3_runner.py --screen
            ├── Screener scores ~103 watchlist tickers (free, local)
            ├── Assembles TickerSnapshot: technicals + fundamentals + FinBERT sentiment
            ├── Runs TradingAgents bull/bear reasoning on top 5 tickers (Anthropic API)
            └── Logs each decision to ~/.tradingagents/signal_log.db

python3 -m logger.score   (run any time)
    └── Fetches price 1d/3d/5d after each signal, computes hit rate + avg return
```

---

## Step 1 — Check Python version

```bash
python --version
python3 --version
```

**Need: 3.12 or higher.** TradingAgents requires it.

If the output is below 3.12, install via pyenv:

```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to shell — append these lines to ~/.bashrc (or ~/.zshrc if using zsh)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Reload shell
source ~/.bashrc

# Install Python 3.12 (takes a few minutes — compiles from source)
pyenv install 3.12.7
pyenv global 3.12.7

# Verify
python --version   # should print Python 3.12.7
```

If Python 3.12+ is already installed, skip to Step 2.

---

## Step 2 — Clone the repo

```bash
cd ~
git clone https://github.com/M4SamCoop/StockTradingAutomation.git
cd StockTradingAutomation
```

Verify the key files are present:

```bash
ls phase3_runner.py run_pipeline.py logger/signal_log.py logger/score.py
```

Expected: all four files listed without error.

---

## Step 3 — Create virtual environment

All subsequent commands run from inside the repo root (`~/StockTradingAutomation`).

```bash
python -m venv venv
source venv/bin/activate

# Verify the venv python is 3.12+
python --version
```

---

## Step 4 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs everything including TradingAgents and its LangGraph/LangChain chain.
Takes 3–5 minutes on first run. FinBERT (~420 MB) downloads on first pipeline run, not here.

Verify key packages loaded correctly:

```bash
python -c "import tradingagents; import ta; import yfinance; import torch; print('OK')"
```

Expected output: `OK`

If `tradingagents` fails with a Python version error, recheck Step 1.

---

## Step 5 — Configure environment variables

```bash
cp .env.example .env
```

If `.env.example` does not exist, create `.env` from scratch:

```bash
cat > .env << 'EOF'
ANTHROPIC_API_KEY=your_key_here
# ALPACA_API_KEY=      # Phase 5 — leave commented for now
# ALPACA_SECRET_KEY=   # Phase 5
# ALPACA_BASE_URL=https://paper-api.alpaca.markets
EOF
```

Now open `.env` and replace `your_key_here` with the actual Anthropic API key:

```bash
nano .env
```

Verify the key is set (this does NOT print the key value):

```bash
python -c "
from dotenv import load_dotenv; import os; load_dotenv()
key = os.getenv('ANTHROPIC_API_KEY', '')
print('Key present:', bool(key) and key != 'your_key_here')
print('Key length:', len(key))
"
```

Expected: `Key present: True` and a length of 100+.

---

## Step 6 — Create log and data directories

```bash
mkdir -p ~/.tradingagents/logs
mkdir -p ~/.tradingagents/cache
mkdir -p ~/.tradingagents/results
```

---

## Step 7 — First manual test run

Run the pipeline manually once to verify everything works end-to-end before
automating it.

```bash
source venv/bin/activate
python phase3_runner.py --tickers AAPL --date $(date +%Y-%m-%d)
```

**What to expect:**
- First run downloads FinBERT model (~420 MB) — takes 1–3 minutes, shows a progress bar
- Subsequent runs skip the download
- Output ends with a DECISIONS table showing AAPL with action + conviction bar
- A signal is logged: `Logged as signal #1`
- Final line shows the DB path: `Signal log: /home/YOUR_USER/.tradingagents/signal_log.db`

If the run fails, check:

```bash
# API key not loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('ANTHROPIC_API_KEY', 'MISSING')[:10])"

# TradingAgents can't find Anthropic client
pip show anthropic
```

---

## Step 8 — Test the screener

Verify the screener can score the full watchlist and pick tickers:

```bash
python phase3_runner.py --screen --shortlist-size 3 --date $(date +%Y-%m-%d)
```

Expected: scores 103 tickers, selects 3, runs reasoning on all 3, logs 3 signals.
Takes 5–15 minutes depending on LLM response time.

---

## Step 9 — Create the daily run script

```bash
cat > scripts/daily_run.sh << 'SCRIPT'
#!/bin/bash
# Daily pre-market signal run — called by cron every weekday morning

set -euo pipefail

REPO_DIR="$HOME/StockTradingAutomation"
LOG_DIR="$HOME/.tradingagents/logs"
DATE=$(date +%Y-%m-%d)
LOGFILE="$LOG_DIR/daily_${DATE}.log"

echo "=== Run started at $(date) ===" >> "$LOGFILE"

cd "$REPO_DIR"
source venv/bin/activate

python phase3_runner.py \
    --screen \
    --shortlist-size 5 \
    --date "$DATE" \
    >> "$LOGFILE" 2>&1

EXIT_CODE=$?
echo "=== Run finished at $(date) — exit code $EXIT_CODE ===" >> "$LOGFILE"
exit $EXIT_CODE
SCRIPT

mkdir -p scripts
chmod +x scripts/daily_run.sh
```

Verify the script runs manually:

```bash
bash scripts/daily_run.sh
cat ~/.tradingagents/logs/daily_$(date +%Y-%m-%d).log
```

Expected: log file contains the full pipeline output and ends with exit code 0.

---

## Step 10 — Set up the cron job

First, determine your UTC offset so the job fires at 8:30 AM US Eastern time
(13:30 UTC in winter / 12:30 UTC in summer during US Daylight Saving Time):

```bash
# Check your current timezone
timedatectl | grep "Time zone"
date +%Z
```

**Cron times by timezone for 8:30 AM ET:**

| Your timezone | Winter (EST, UTC-5) | Summer (EDT, UTC-4) |
|---|---|---|
| UTC | 13:30 | 12:30 |
| BST / CET+0 | 13:30 | 12:30 |
| CET (UTC+1) | 14:30 | 13:30 |
| EET (UTC+2) | 15:30 | 14:30 |
| MSK (UTC+3) | 16:30 | 15:30 |

US Daylight Saving runs March–November. The simplest approach is to set the cron
time for your local clock at the equivalent of 8:30 AM ET, and adjust it twice a year.

Open the crontab editor:

```bash
crontab -e
```

Add this line (adjust the hour/minute for your timezone using the table above):

```
# Trading signal pipeline — weekdays at 8:30 AM ET (adjust hour for your timezone)
30 13 * * 1-5 /home/YOUR_USER/StockTradingAutomation/scripts/daily_run.sh
```

**Replace `YOUR_USER` with your actual username.** Get it with:

```bash
echo $USER
```

Save and exit the editor. Verify the cron entry was saved:

```bash
crontab -l
```

Expected: the line you added is shown.

---

## Step 11 — Verify cron can find the environment

Cron runs in a stripped environment — it won't have the same PATH as your shell.
Test this by running the script the same way cron does:

```bash
env -i HOME=$HOME /bin/bash /home/$USER/StockTradingAutomation/scripts/daily_run.sh
```

If this fails with a `python not found` or `venv not found` error, use an absolute
path to the venv python in the script. Edit `scripts/daily_run.sh` and change:

```bash
source venv/bin/activate
python phase3_runner.py \
```

to:

```bash
/home/YOUR_USER/StockTradingAutomation/venv/bin/python phase3_runner.py \
```

---

## Step 12 — Check logs and signal database

**Check today's run log:**

```bash
cat ~/.tradingagents/logs/daily_$(date +%Y-%m-%d).log
```

**Check all logged signals:**

```bash
cd ~/StockTradingAutomation
source venv/bin/activate
python -c "
from logger.signal_log import get_all_signals
signals = get_all_signals()
print(f'{len(signals)} signals logged')
for s in signals[-10:]:
    print(f'  {s[\"signal_date\"]}  {s[\"ticker\"]:<6}  {s[\"action\"]:<4}  {s[\"conviction\"]:.0%}  entry=\${s[\"entry_price\"]}')
"
```

**Score outcomes (run any time — needs signals at least 1 day old):**

```bash
python -m logger.score
```

**Score report only (no new fetches):**

```bash
python -m logger.score --report
```

---

## Day-to-day reference

```bash
# Activate venv (required before any manual command)
cd ~/StockTradingAutomation && source venv/bin/activate

# Manual run on specific tickers (no screener)
python phase3_runner.py --tickers AAPL MSFT NVDA

# Manual run with screener
python phase3_runner.py --screen --shortlist-size 5

# Run without logging to DB (dry run)
python phase3_runner.py --tickers AAPL --no-log

# Check today's cron log
cat ~/.tradingagents/logs/daily_$(date +%Y-%m-%d).log

# Tail the log live during a run
tail -f ~/.tradingagents/logs/daily_$(date +%Y-%m-%d).log

# Score all aged signals + print report
python -m logger.score

# Print report only
python -m logger.score --report

# Check cron is still scheduled
crontab -l
```

---

## Expected state after two weeks

After 10 trading days of automated runs you should have:

- 10 log files in `~/.tradingagents/logs/`
- 40–50 signals in `~/.tradingagents/signal_log.db` (5 tickers × 10 days)
- `python -m logger.score` showing hit rates for +1d, +3d, and +5d horizons
- Enough data to decide whether to proceed to paper trading

**Signal quality threshold to proceed to paper trading:**
- BUY hit rate ≥ 55% at +1d horizon across at least 20 directional signals
- High-conviction signals (≥ 0.75) outperforming low-conviction ones
- No systematic failures (all HOLD, errors, or zero BUY/SELL signals)

If results look good after two weeks, the next step is `SETUP_PAPER_TRADING.md`
(not yet written — will be added when ready to build Phase 5).
