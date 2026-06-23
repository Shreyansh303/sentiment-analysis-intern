#!/bin/zsh
# run_daily.sh — daily sentiment pipeline, intended to run via cron on weekday
# mornings. Fetches fresh news, scores it, and updates the live backtest record.
# Each step is best-effort: if one fails (e.g. a network blip on the fetch),
# the script logs the error and still runs the rest.
#
# NOTE: fetch_articles.py refreshes (overwrites) articles_fetched.json by design
# — that is the "live news of the day" file the pipeline scores.

PROJECT="$HOME/Downloads/sentiment analysis"
cd "$PROJECT" || exit 1

# cron runs with a bare environment, so make pyenv's python (the one with
# FinBERT / scipy / yfinance installed) resolve exactly as it does in your shell.
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:/usr/local/bin:/usr/bin:/bin"
if command -v pyenv >/dev/null 2>&1; then eval "$(pyenv init -)"; fi

# Use the already-downloaded FinBERT model: no re-download, no HF warning.
export HF_HUB_OFFLINE=1

mkdir -p "$PROJECT/logs"
LOG="$PROJECT/logs/$(date +%Y-%m-%d)-daily.log"

{
  echo "================ run started $(date) ================"
  echo "python3 -> $(command -v python3)"

  echo "--- step 1: fetch fresh news (fetch_articles.py) ---"
  python3 fetch_articles.py

  echo "--- step 2: score + snapshot + DB write (pipeline_nifty.py) ---"
  python3 pipeline_nifty.py

  echo "--- step 3: update live backtest record (backtest.py) ---"
  python3 backtest.py

  echo "================ run finished $(date) ================"
  echo
} >> "$LOG" 2>&1
