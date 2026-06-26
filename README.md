# Nifty News-Sentiment Signal Research

A research pipeline that scores financial news and earnings transcripts for
Indian large-cap stocks (Nifty 50) and tests, **without lookahead leakage**,
whether the resulting sentiment signal has any predictive power over forward
stock returns.

The headline finding is honest and negative — see [Results](#results). The point
of this repo is the *methodology*: a leakage-disciplined way to build, score,
falsify, and then forward-track a sentiment alpha signal end to end.

## What it does

1. **Ingest** — pull a universe of Nifty news + earnings articles
   (`fetch_universe.py`, `fetch_articles.py`, `fetch_earnings.py`).
2. **Score** — run each article through **FinBERT** (transformer) and a
   **Loughran–McDonald** finance lexicon, emitting raw pos/neg/neu class
   probabilities (`sentiment_core.py`). This expensive pass runs **once**.
3. **Aggregate** — cheaply re-weight those raw probabilities into per-ticker
   sentiment/news aggregates and labels for any given parameter set, which makes
   the parameter sensitivity sweep tractable (`research/sensitivity.py`).
4. **Validate** — two independent, leakage-free evaluation harnesses:
   - `backtest.py` — joins each `(date, ticker)` signal to its **forward**
     return (acted at the *next* session's open), and reports Information
     Coefficient (Spearman), directional hit-rate, and a toy long-short P&L net
     of cost.
   - `evaluate_cutoff.py` — a single as-of-date experiment with an explicit
     information barrier: the model sees only articles dated on/before a hard
     cutoff, predicts UP/DOWN/NEUTRAL, then is scored against the actual price
     path after the cutoff.
5. **Forward-track** — `run_daily.sh` runs the live pipeline on a schedule
   (fetch → score → snapshot → backtest), accumulating a real out-of-sample
   record one trading day at a time rather than re-slicing history.

Data persists to a versioned SQLite store (`db/`, `db_io.py`), which is the
source of truth. Every scoring and snapshot step **dual-writes** to both the DB
and committed CSVs (`snapshots.csv`, `scored_history.csv`), so the analysis
panel survives even if the CSVs are regenerated.

## Design notes worth a look

- **Score once / re-weight many.** The transformer pass writes raw class
  probabilities so every downstream parameter choice is a cheap derivation, not
  a re-inference. This is what makes a full sensitivity sweep feasible.
- **Leakage is enforced, not assumed.** A snapshot dated `D` only uses news
  through `D`; returns are measured open-to-open starting the *next* session, so
  the signal date never touches its own return window. `evaluate_cutoff.py` adds
  hard assertions that no post-cutoff byte reaches the model.
- **DB is the source of truth; CSVs are a mirror.** The daily run upserts by
  `(date, ticker)` so re-runs overwrite rather than duplicate, and the committed
  CSVs can be rebuilt from the DB at any time.
- **Every magic number is a swept-able `Config` default** (`sentiment_core.py`),
  so robustness can be tested rather than hand-tuned.

## Results

Across the validated experiments, no signal clears the bar for a tradeable edge.
Reporting this honestly is the point: the value here is a falsification harness
that can kill a bad signal cleanly, not a claim of alpha.

- **Cross-sectional news sentiment** (`run_universe_all.py` — 11 biweekly windows
  over the Nifty 50, Jan–Jun 2026): mean rank-IC **+0.05** (p≈0.13, ~54%
  directional accuracy). Positive but not significant, and it does **not** beat a
  price-momentum baseline (+0.08).
- **Earnings-tone surprise** (QoQ delta — `earnings_surprise.py`,
  `oos_earnings_surprise.py`): the strongest in-sample candidate at the 20-day
  horizon (mean IC **+0.17**, p≈0.02), but it **fails the out-of-sample gate** —
  IC decays ~**70%** train→test and the long-short spread flips negative on the
  held-out quarters. Classic in-sample inflation.

The honest read: earnings-tone surprise is the best remaining candidate, but it
needs live forward quarters — not more historical slicing — to resolve. The
daily harness (`run_daily.sh`) now accumulates exactly that forward record.

## Layout

| Path | Purpose |
|------|---------|
| `sentiment_core.py` | FinBERT + Loughran–McDonald scoring, `Config`, aggregation |
| `universe.py`, `db_io.py` | Universe definition, SQLite I/O layer (source of truth) |
| `pipeline_nifty.py` | Daily end-to-end scoring run (dual-writes DB + CSV) |
| `run_daily.sh` | Cron-friendly daily driver: fetch → score → snapshot → backtest |
| `run_universe_all.py` | Nifty-50 cross-sectional biweekly IC / long-short sweep |
| `backtest.py` | Forward-return validation harness (IC, hit-rate, P&L) |
| `evaluate_cutoff.py` | Leakage-free as-of-date experiment |
| `earnings_surprise.py`, `oos_earnings_surprise.py` | Earnings-tone QoQ surprise test + out-of-sample gate |
| `pick_top.py`, `grade_ledger.py`, `pick_backtest.py` | Long-only pick list, reason cards, outcome grading + OOS pick gate |
| `fetch_*.py` | News, earnings, universe ingestion |
| `export_jan_from_db.py`, `check_setup.py` | Utilities: DB article export, environment check |
| `snapshots.csv`, `scored_history.csv` | Accumulating daily panel + raw scores (mirror of DB) |
| `research/` | Exploratory scripts: sensitivity sweep, signal search, earnings drift, news-leg experiment, combined book |
| `results/` | Committed output CSVs (scored article caches, backtest results, decision ledger) |
| `db/` | Versioned SQLite schema and migrations |
