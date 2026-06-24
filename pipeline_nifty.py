"""
Sentiment vs News Analyst — full Nifty 50 universe.
Scopes: ticker-specific, SECTOR_IT (propagates to TCS+INFY), MACRO (propagates to all).

All tunable numbers now live in sentiment_core.Config; this script just runs the
default config, prints the per-article + per-ticker tables (now with bootstrap CI
bands and low-n flags), and appends today's aggregates to snapshots.csv.
"""
import os
from datetime import date

import numpy as np
import pandas as pd

from sentiment_core import (
    Config, SECTOR_OF, load_models, score_articles, apply_weights, aggregate,
    save_scored_history,
)
from universe import TICKERS   # full Nifty 50, consistent with run_universe_all

TODAY = date.today()   # dynamic, so future-dated articles can't get recency weight > 1.0
CFG = Config()


def resolve_articles_file():
    for p in ("articles_fetched.json", "files/articles_fetched.json", "../articles_fetched.json"):
        if os.path.exists(p):
            print(f"Using live news file: {p}")
            return p
    fallback = "articles_nifty.json"
    if not os.path.exists(fallback) and os.path.exists("files/articles_nifty.json"):
        fallback = "files/articles_nifty.json"
    print(f"Live news file 'articles_fetched.json' not found. Falling back to: {fallback}")
    return fallback


ARTICLES_FILE = resolve_articles_file()

# 1. Load + score articles ONCE (expensive FinBERT/LM pass), then weight with CFG.
df = pd.read_json(ARTICLES_FILE)
# Mirror the loaded articles into the DB first, so the scored-history rows
# written below satisfy their foreign key to the articles table.
try:
    import db_io
    db_io.upsert_articles(df.to_dict("records"), source_batch="live")
except Exception as e:
    print(f"  (DB) articles mirror skipped: {e}")
df = score_articles(df, models=load_models())
# Persist raw scores so sensitivity.py can replay the sweep over real history.
save_scored_history(df, TODAY)
df = apply_weights(df, CFG, TODAY)

df["why_sent"] = [f"FinBERT mood (pos: {p:.2f}, neg: {n:.2f}, neu: {u:.2f})."
                  for p, n, u in zip(df["pos_prob"], df["neg_prob"], df["neu_prob"])]
df["why_news"] = [f"LM Lexicon (pos words: {p}, neg words: {n})."
                  for p, n in zip(df["pos_count"], df["neg_count"])]

# 2. Per-article table
print(f"{'ID':<11}{'Scope':<13}{'Date':<12}{'Sent':>6}{'w_sent':>8}{'cut_s':>6}{'News':>6}{'w_news':>8}{'cut_n':>6}  Impact")
print("-" * 110)
for _, row in df.iterrows():
    flag_sent = "CUT" if row["cut_sent"] else "   "
    flag_news = "CUT" if row["cut_news"] else "   "
    a_id = str(row["id"])[:10]
    date_str = str(row["date"])[:10]
    print(f"{a_id:<11}{row['scope']:<13}{date_str:<12}{row['sent']:>6.2f}{row['w_sent']:>8.3f}{flag_sent:>6} "
          f"{row['news']:>6.2f}{row['w_news']:>8.3f}{flag_news:>6}  {row['impact']:<5}")
    print(f"     ├─ Sentiment: {row['why_sent']}")
    print(f"     └─ News:      {row['why_news']}")

# 3. Per-ticker aggregates with bootstrap CI bands + low-n flags
print("\n=== PER-TICKER AGGREGATES (Independent Sentiment / News Analysts) ===")
print("    CI = 95% bootstrap band on the weighted mean; (!) flags low-n cells "
      f"(< {CFG.low_n_threshold} articles).")
snapshot_rows = aggregate(df, CFG, TODAY, tickers=TICKERS)
for r in snapshot_rows:
    warn = " (!) LOW-N" if r["low_n"] else ""
    print(
        f"{r['ticker']:<13} "
        f"sentiment={r['agg_sent']:+.2f} [{r['sent_lo']:+.2f},{r['sent_hi']:+.2f}] (n={r['n_sent']:<2}) | "
        f"news={r['agg_news']:+.2f} [{r['news_lo']:+.2f},{r['news_hi']:+.2f}] (n={r['n_news']:<2}){warn}"
    )

# 4. Append today's aggregates to the daily panel (idempotent per date).
#    This is the time series the backtest harness validates against — every
#    missed day is lost history, so we persist on every run.
SNAPSHOT_FILE = "snapshots.csv"
snap_today = pd.DataFrame(snapshot_rows)

if os.path.exists(SNAPSHOT_FILE):
    panel = pd.read_csv(SNAPSHOT_FILE)
    panel = panel[panel["date"] != TODAY.isoformat()]  # re-runs overwrite, not duplicate
    panel = pd.concat([panel, snap_today], ignore_index=True)
else:
    panel = snap_today

panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
panel.to_csv(SNAPSHOT_FILE, index=False)

# Dual-write: upsert today's aggregates into the database (snapshots table).
try:
    import db_io
    if db_io.upsert_snapshots(snap_today):
        print(f"  (DB) upserted {len(snap_today)} rows into snapshots table.")
except Exception as e:
    print(f"  (DB) snapshots write skipped: {e}")

print(f"\nAppended {len(snap_today)} rows to {SNAPSHOT_FILE} "
      f"(panel now {len(panel)} rows across {panel['date'].nunique()} day(s)).")
