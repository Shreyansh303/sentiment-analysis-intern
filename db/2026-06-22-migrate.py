#!/usr/bin/env python3
"""
Migrate the sentiment-analysis CSV/JSON files into a relational SQLite database.

Design choices:
  * Idempotent: re-running does not duplicate rows. Every load uses
    INSERT .. ON CONFLICT DO UPDATE keyed on the natural primary key, so
    re-running the upstream pipeline and re-loading is safe.
  * Transactional: each table load runs inside a single transaction; a
    failure rolls the whole table back rather than leaving a partial load.
  * Typed: dates -> ISO DATE strings, probabilities -> float, counts -> int,
    booleans -> 0/1. Empty strings become NULL.
  * Portable: the schema (2026-06-22-schema.sql) is Postgres-compatible.
    To target Postgres instead, swap sqlite3 for psycopg and change the
    upsert placeholder style (see notes at bottom).

Usage:
    python3 2026-06-22-migrate.py                # builds ./sentiment.db
    python3 2026-06-22-migrate.py --db out.db    # custom output path
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import sqlite3
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)            # parent folder holds the source files
SCHEMA = os.path.join(HERE, "2026-06-22-schema.sql")


# --------------------------- type coercion helpers ---------------------------
def f(v):
    """Float or None."""
    if v is None or v == "":
        return None
    return float(v)


def i(v):
    """Int or None."""
    if v is None or v == "":
        return None
    return int(float(v))


def b(v):
    """Bool (0/1) or None."""
    if v is None or v == "":
        return None
    return 1 if str(v).strip().lower() in ("true", "1", "yes") else 0


def d(v):
    """Normalize a date-ish string to ISO YYYY-MM-DD, or None."""
    if v is None or v == "":
        return None
    v = str(v).strip()
    # source dates are already ISO (YYYY-MM-DD); pass through, validate.
    try:
        date.fromisoformat(v[:10])
        return v[:10]
    except ValueError:
        return v  # keep raw if non-standard; constraint will surface issues


def s(v):
    """String or None (empty -> None)."""
    if v is None:
        return None
    v = str(v)
    return v if v != "" else None


# --------------------------------- loaders -----------------------------------
def load_articles(con):
    """Load + dedup articles from all article_*.json files (id is the key)."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "articles_*.json")))
    rows, seen = [], set()
    raw = 0
    for path in files:
        with open(path) as fh:
            for r in json.load(fh):
                raw += 1
                if r["id"] in seen:
                    continue            # in-batch dedup; upsert handles cross-run
                seen.add(r["id"])
                rows.append((
                    r["id"], r["scope"], r["source"],
                    d(r["date"]), s(r.get("url")), r["text"],
                ))
    sql = """
        INSERT INTO articles (id, ticker, source, published_date, url, text)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            ticker=excluded.ticker, source=excluded.source,
            published_date=excluded.published_date, url=excluded.url,
            text=excluded.text
    """
    with con:
        con.executemany(sql, rows)
    print(f"  articles: {raw} raw records -> {len(rows)} unique loaded "
          f"(from {len(files)} files)")


def load_article_scores(con):
    path = os.path.join(DATA_DIR, "scored_history.csv")
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append((
                r["id"], d(r["snap_date"]), r["scope"], s(r["source"]),
                d(r["date"]), f(r["pos_prob"]), f(r["neg_prob"]),
                f(r["neu_prob"]), f(r["sent"]), f(r["news"]),
                i(r["pos_count"]), i(r["neg_count"]),
            ))
    sql = """
        INSERT INTO article_scores
            (id, snap_date, scope, source, article_date, pos_prob, neg_prob,
             neu_prob, sent, news, pos_count, neg_count)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id, snap_date) DO UPDATE SET
            scope=excluded.scope, source=excluded.source,
            article_date=excluded.article_date, pos_prob=excluded.pos_prob,
            neg_prob=excluded.neg_prob, neu_prob=excluded.neu_prob,
            sent=excluded.sent, news=excluded.news,
            pos_count=excluded.pos_count, neg_count=excluded.neg_count
    """
    with con:
        con.executemany(sql, rows)
    print(f"  article_scores: {len(rows)} rows loaded")


def _signal_rows(path, extra=False):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            base = [
                d(r["date"]), r["ticker"], f(r["agg_sent"]), f(r["agg_news"]),
                f(r["divergence"]), s(r["label"]), i(r["n_sent"]), i(r["n_news"]),
                f(r["sent_lo"]), f(r["sent_hi"]), f(r["news_lo"]), f(r["news_hi"]),
                b(r["low_n"]),
            ]
            if extra:
                base += [
                    f(r["combined"]), f(r["actual_ret"]), f(r["momentum"]),
                    s(r["window"]), f(r["ret_resid"]), f(r["sig_resid"]),
                ]
            rows.append(tuple(base))
    return rows


def load_snapshots(con):
    rows = _signal_rows(os.path.join(DATA_DIR, "snapshots.csv"))
    sql = """
        INSERT INTO snapshots
            (date, ticker, agg_sent, agg_news, divergence, label, n_sent,
             n_news, sent_lo, sent_hi, news_lo, news_hi, low_n)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(date, ticker) DO UPDATE SET
            agg_sent=excluded.agg_sent, agg_news=excluded.agg_news,
            divergence=excluded.divergence, label=excluded.label,
            n_sent=excluded.n_sent, n_news=excluded.n_news,
            sent_lo=excluded.sent_lo, sent_hi=excluded.sent_hi,
            news_lo=excluded.news_lo, news_hi=excluded.news_hi,
            low_n=excluded.low_n
    """
    with con:
        con.executemany(sql, rows)
    print(f"  snapshots: {len(rows)} rows loaded")


def load_universe_calls(con):
    rows = _signal_rows(os.path.join(DATA_DIR, "universe_calls.csv"), extra=True)
    sql = """
        INSERT INTO universe_calls
            (date, ticker, agg_sent, agg_news, divergence, label, n_sent,
             n_news, sent_lo, sent_hi, news_lo, news_hi, low_n,
             combined, actual_ret, momentum, window, ret_resid, sig_resid)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(date, ticker) DO UPDATE SET
            agg_sent=excluded.agg_sent, agg_news=excluded.agg_news,
            divergence=excluded.divergence, label=excluded.label,
            n_sent=excluded.n_sent, n_news=excluded.n_news,
            sent_lo=excluded.sent_lo, sent_hi=excluded.sent_hi,
            news_lo=excluded.news_lo, news_hi=excluded.news_hi, low_n=excluded.low_n,
            combined=excluded.combined, actual_ret=excluded.actual_ret,
            momentum=excluded.momentum, window=excluded.window,
            ret_resid=excluded.ret_resid, sig_resid=excluded.sig_resid
    """
    with con:
        con.executemany(sql, rows)
    print(f"  universe_calls: {len(rows)} rows loaded")


def load_backtest_pnl(con):
    path = os.path.join(DATA_DIR, "combined_book_pnl.csv")
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append((s(r["window"]), f(r["ret"]), f(r["turnover"])))
    sql = """
        INSERT INTO backtest_pnl (window, ret, turnover)
        VALUES (?, ?, ?)
        ON CONFLICT(window) DO UPDATE SET
            ret=excluded.ret, turnover=excluded.turnover
    """
    with con:
        con.executemany(sql, rows)
    print(f"  backtest_pnl: {len(rows)} rows loaded")


# ----------------------------------- main ------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(HERE, "sentiment.db"))
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.execute("PRAGMA foreign_keys = ON;")   # enforce FK integrity

    print(f"Applying schema -> {args.db}")
    with open(SCHEMA) as fh:
        con.executescript(fh.read())

    print("Loading tables:")
    load_articles(con)          # parents before children (FK order)
    load_article_scores(con)
    load_snapshots(con)
    load_universe_calls(con)
    load_backtest_pnl(con)

    con.close()
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())

# -----------------------------------------------------------------------------
# Promoting to PostgreSQL later:
#   1. Run 2026-06-22-schema.sql against your Postgres database unchanged.
#   2. Replace `import sqlite3` / connect with psycopg:
#         import psycopg
#         con = psycopg.connect("postgresql://user:pass@host/dbname")
#   3. Change every "?" placeholder to "%s" and
#      "ON CONFLICT(col) DO UPDATE SET x=excluded.x" stays valid in Postgres
#      (Postgres also supports the EXCLUDED keyword).
#   4. Drop the PRAGMA line (Postgres enforces FKs by default).
# -----------------------------------------------------------------------------
