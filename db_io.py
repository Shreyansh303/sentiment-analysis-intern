"""
db_io.py — the single doorway between the pipeline scripts and the SQLite
database (db/sentiment.db).

Why this exists
---------------
The scripts used to read and write loose CSV/JSON files. This module gives them
drop-in functions that read from / write to the database instead, while the
scripts KEEP writing their CSV/JSON files too (dual-write) as a safety net during
the transition. Each write here is idempotent: re-running a script refreshes rows
rather than duplicating them.

Design notes
------------
* Connection: one helper, foreign keys enabled.
* DB location: db/sentiment.db next to this file. Override with the
  SENTIMENT_DB environment variable.
* Robustness: write_* helpers are best-effort — if the DB write fails they print
  a warning and return False, so a database hiccup never takes down a pipeline
  run that is still writing its CSV/JSON files. read_* helpers return None on
  failure so callers can fall back to their CSV.
* Schema: ensure_schema() creates the v2 tables and the articles.source_batch
  column on first use, so scripts work even if the SQL files were never run.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import date

import pandas as pd

# --------------------------------------------------------------------------- #
# Connection / location
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("SENTIMENT_DB", os.path.join(_HERE, "db", "sentiment.db"))

# Column orders matching the source files (so DB round-trips look identical).
_ARTICLE_COLS = ["id", "ticker", "source", "published_date", "url", "text", "source_batch"]
_SCORED_COLS = ["id", "scope", "source", "date", "pos_prob", "neg_prob",
                "neu_prob", "sent", "news", "pos_count", "neg_count"]
_SNAP_COLS = ["date", "ticker", "agg_sent", "agg_news", "divergence", "label",
              "n_sent", "n_news", "sent_lo", "sent_hi", "news_lo", "news_hi", "low_n"]
_UNIVERSE_COLS = ["date", "ticker", "agg_sent", "agg_news", "divergence", "label",
                  "n_sent", "n_news", "sent_lo", "sent_hi", "news_lo", "news_hi",
                  "low_n", "combined", "actual_ret", "momentum", "window",
                  "ret_resid", "sig_resid", "agg_earnings"]
_EARNINGS_COLS = ["ticker", "quarter", "quarter_end", "transcript_date",
                  "finbert_sent", "lm_polarity", "lm_uncertainty", "lm_modal",
                  "n_chunks", "source", "agg_earnings"]
_UNI_WIN_COLS = ["window", "n", "ic_sent", "ic_mom", "ls_sent", "ls_mom"]
_PNL_RUN_COLS = ["signal", "horizon", "date", "ret", "cum"]
_CUTOFF_COLS = ["cutoff", "eval_end", "ticker", "agg_sent", "agg_news", "combined",
                "label", "prediction", "n_sent", "n_news", "low_n", "entry_date",
                "exit_date", "entry_px", "exit_px", "actual_ret", "actual_dir"]


def get_conn(path: str | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(path or DB_PATH)
    con.execute("PRAGMA foreign_keys = ON;")
    return con


# --------------------------------------------------------------------------- #
# Schema bootstrap (idempotent)
# --------------------------------------------------------------------------- #
_V2_DDL = """
CREATE TABLE IF NOT EXISTS earnings_scores (
    ticker          TEXT    NOT NULL,
    quarter         TEXT    NOT NULL,
    quarter_end     DATE,
    transcript_date DATE,
    finbert_sent    NUMERIC,
    lm_polarity     NUMERIC,
    lm_uncertainty  NUMERIC,
    lm_modal        NUMERIC,
    n_chunks        INTEGER,
    source          TEXT,
    agg_earnings    NUMERIC,
    PRIMARY KEY (ticker, quarter)
);
CREATE TABLE IF NOT EXISTS universe_windows (
    window TEXT PRIMARY KEY, n INTEGER, ic_sent NUMERIC, ic_mom NUMERIC,
    ls_sent NUMERIC, ls_mom NUMERIC
);
CREATE TABLE IF NOT EXISTS backtest_pnl_runs (
    signal TEXT NOT NULL, horizon TEXT NOT NULL, date DATE NOT NULL,
    ret NUMERIC, cum NUMERIC, PRIMARY KEY (signal, horizon, date)
);
CREATE TABLE IF NOT EXISTS cutoff_eval (
    cutoff DATE NOT NULL, eval_end DATE NOT NULL, ticker TEXT NOT NULL,
    agg_sent NUMERIC, agg_news NUMERIC, combined NUMERIC, label TEXT,
    prediction TEXT, n_sent INTEGER, n_news INTEGER, low_n BOOLEAN,
    entry_date DATE, exit_date DATE, entry_px NUMERIC, exit_px NUMERIC,
    actual_ret NUMERIC, actual_dir TEXT,
    PRIMARY KEY (cutoff, eval_end, ticker)
);
"""


def ensure_schema(con: sqlite3.Connection) -> None:
    """Create v2 tables and the articles.source_batch column if missing."""
    con.executescript(_V2_DDL)
    # ALTER ADD COLUMN has no IF NOT EXISTS in SQLite — try and ignore if present.
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(articles)").fetchall()]
        if cols and "source_batch" not in cols:
            con.execute("ALTER TABLE articles ADD COLUMN source_batch TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(universe_calls)").fetchall()]
        if cols and "agg_earnings" not in cols:
            con.execute("ALTER TABLE universe_calls ADD COLUMN agg_earnings NUMERIC")
    except sqlite3.OperationalError:
        pass
    con.commit()


def _align(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return df restricted/reordered to `cols`, adding any missing as None."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


# --------------------------------------------------------------------------- #
# articles  (incremental upsert on id)
# --------------------------------------------------------------------------- #
def upsert_articles(records, source_batch: str | None = None) -> bool:
    """records: list of dicts (id, scope, source, date, text, url) — the fetcher
    schema. Maps scope->ticker, date->published_date and upserts on id."""
    try:
        rows = []
        for r in records:
            _d = r.get("date")
            rows.append((
                r["id"], r.get("scope"), r.get("source"),
                (str(_d)[:10] if _d is not None else None),
                r.get("url"), r.get("text"), source_batch,
            ))
        con = get_conn()
        ensure_schema(con)
        con.executemany(
            """INSERT INTO articles
                 (id, ticker, source, published_date, url, text, source_batch)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 ticker=excluded.ticker, source=excluded.source,
                 published_date=excluded.published_date, url=excluded.url,
                 text=excluded.text, source_batch=excluded.source_batch""",
            rows,
        )
        con.commit(); con.close()
        return True
    except Exception as e:  # best-effort: never break the file write path
        print(f"[db_io] WARN upsert_articles failed: {e}")
        return False


def read_articles(source_batch: str | None = None) -> pd.DataFrame | None:
    """Return articles in the fetcher schema (id, scope, source, date, text, url).
    Optionally filter to one source_batch. None on failure."""
    try:
        con = get_conn()
        q = ("SELECT id, ticker AS scope, source, published_date AS date, "
             "text, url FROM articles")
        params = ()
        if source_batch is not None:
            q += " WHERE source_batch = ?"; params = (source_batch,)
        df = pd.read_sql_query(q, con, params=params)
        con.close()
        return df
    except Exception as e:
        print(f"[db_io] WARN read_articles failed: {e}")
        return None


# --------------------------------------------------------------------------- #
# article_scores  (scored_history; upsert on id+snap_date)
# --------------------------------------------------------------------------- #
def upsert_article_scores(df: pd.DataFrame, snap_date: date) -> bool:
    """df holds the _SCORED_COLS columns (id, scope, source, date, probs...).
    Stored keyed by (id, snap_date); the source 'date' maps to article_date."""
    try:
        rec = df[_SCORED_COLS].copy()
        snap = snap_date.isoformat() if hasattr(snap_date, "isoformat") else str(snap_date)
        rows = []
        for _, r in rec.iterrows():
            rows.append((
                r["id"], snap, r["scope"], r["source"], str(r["date"])[:10],
                _num(r["pos_prob"]), _num(r["neg_prob"]), _num(r["neu_prob"]),
                _num(r["sent"]), _num(r["news"]), _int(r["pos_count"]), _int(r["neg_count"]),
            ))
        con = get_conn()
        con.executemany(
            """INSERT INTO article_scores
                 (id, snap_date, scope, source, article_date, pos_prob, neg_prob,
                  neu_prob, sent, news, pos_count, neg_count)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id, snap_date) DO UPDATE SET
                 scope=excluded.scope, source=excluded.source,
                 article_date=excluded.article_date, pos_prob=excluded.pos_prob,
                 neg_prob=excluded.neg_prob, neu_prob=excluded.neu_prob,
                 sent=excluded.sent, news=excluded.news,
                 pos_count=excluded.pos_count, neg_count=excluded.neg_count""",
            rows,
        )
        con.commit(); con.close()
        return True
    except Exception as e:
        print(f"[db_io] WARN upsert_article_scores failed: {e}")
        return False


def read_scored_history() -> pd.DataFrame | None:
    """Return the scored history in the CSV schema (snap_date, id, scope, source,
    date, probs...). article_date is renamed back to 'date'. None on failure."""
    try:
        con = get_conn()
        df = pd.read_sql_query(
            """SELECT snap_date, id, scope, source, article_date AS date,
                      pos_prob, neg_prob, neu_prob, sent, news, pos_count, neg_count
               FROM article_scores""", con)
        con.close()
        return df
    except Exception as e:
        print(f"[db_io] WARN read_scored_history failed: {e}")
        return None


# --------------------------------------------------------------------------- #
# snapshots  (daily panel; upsert on date+ticker)
# --------------------------------------------------------------------------- #
def upsert_snapshots(df: pd.DataFrame) -> bool:
    try:
        rec = _align(df, _SNAP_COLS)
        rows = [tuple(_cell(v) for v in row) for row in rec.itertuples(index=False)]
        con = get_conn()
        con.executemany(
            """INSERT INTO snapshots
                 (date, ticker, agg_sent, agg_news, divergence, label, n_sent,
                  n_news, sent_lo, sent_hi, news_lo, news_hi, low_n)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(date, ticker) DO UPDATE SET
                 agg_sent=excluded.agg_sent, agg_news=excluded.agg_news,
                 divergence=excluded.divergence, label=excluded.label,
                 n_sent=excluded.n_sent, n_news=excluded.n_news,
                 sent_lo=excluded.sent_lo, sent_hi=excluded.sent_hi,
                 news_lo=excluded.news_lo, news_hi=excluded.news_hi,
                 low_n=excluded.low_n""",
            rows,
        )
        con.commit(); con.close()
        return True
    except Exception as e:
        print(f"[db_io] WARN upsert_snapshots failed: {e}")
        return False


def read_snapshots() -> pd.DataFrame | None:
    try:
        con = get_conn()
        df = pd.read_sql_query("SELECT * FROM snapshots", con)
        con.close()
        if df is None or df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df
    except Exception as e:
        print(f"[db_io] WARN read_snapshots failed: {e}")
        return None


# --------------------------------------------------------------------------- #
# universe_calls / universe_windows  (full-replace each run)
# --------------------------------------------------------------------------- #
def write_universe_calls(df: pd.DataFrame) -> bool:
    return _replace_table(df, "universe_calls", _UNIVERSE_COLS)


def read_universe_calls() -> pd.DataFrame | None:
    try:
        con = get_conn()
        df = pd.read_sql_query("SELECT * FROM universe_calls", con)
        con.close()
        return df if (df is not None and not df.empty) else None
    except Exception as e:
        print(f"[db_io] WARN read_universe_calls failed: {e}")
        return None


def write_universe_windows(df: pd.DataFrame) -> bool:
    return _replace_table(df, "universe_windows", _UNI_WIN_COLS)


# --------------------------------------------------------------------------- #
# earnings_scores  (quarterly transcript scores; upsert on ticker+quarter)
# --------------------------------------------------------------------------- #
def upsert_earnings_scores(df: pd.DataFrame) -> bool:
    try:
        rec = _align(df, _EARNINGS_COLS)
        rows = [tuple(_cell(v) for v in row) for row in rec.itertuples(index=False)]
        con = get_conn()
        ensure_schema(con)
        con.executemany(
            """INSERT INTO earnings_scores
               (ticker, quarter, quarter_end, transcript_date,
                finbert_sent, lm_polarity, lm_uncertainty, lm_modal,
                n_chunks, source, agg_earnings)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(ticker, quarter) DO UPDATE SET
                 transcript_date=excluded.transcript_date,
                 finbert_sent=excluded.finbert_sent,
                 lm_polarity=excluded.lm_polarity,
                 lm_uncertainty=excluded.lm_uncertainty,
                 lm_modal=excluded.lm_modal,
                 n_chunks=excluded.n_chunks,
                 source=excluded.source,
                 agg_earnings=excluded.agg_earnings""",
            rows,
        )
        con.commit(); con.close()
        return True
    except Exception as e:
        print(f"[db_io] WARN upsert_earnings_scores failed: {e}")
        return False


def read_earnings_scores() -> pd.DataFrame | None:
    try:
        con = get_conn()
        df = pd.read_sql_query("SELECT * FROM earnings_scores", con)
        con.close()
        return df if (df is not None and not df.empty) else None
    except Exception as e:
        print(f"[db_io] WARN read_earnings_scores failed: {e}")
        return None


# --------------------------------------------------------------------------- #
# backtest outputs
# --------------------------------------------------------------------------- #
def write_backtest_pnl(df: pd.DataFrame) -> bool:
    """combined_book_pnl: window, ret, turnover — full replace."""
    return _replace_table(df, "backtest_pnl", ["window", "ret", "turnover"])


def write_backtest_pnl_run(df: pd.DataFrame, signal: str, horizon: str) -> bool:
    """Per-run P&L curve, replacing rows for this (signal, horizon)."""
    try:
        rec = df.copy()
        rec["signal"] = signal
        rec["horizon"] = horizon
        rec["date"] = rec["date"].astype(str).str.slice(0, 10)
        rec = _align(rec, _PNL_RUN_COLS)
        con = get_conn()
        ensure_schema(con)
        con.execute("DELETE FROM backtest_pnl_runs WHERE signal=? AND horizon=?",
                    (signal, horizon))
        rec.to_sql("backtest_pnl_runs", con, if_exists="append", index=False)
        con.commit(); con.close()
        return True
    except Exception as e:
        print(f"[db_io] WARN write_backtest_pnl_run failed: {e}")
        return False


def write_cutoff_eval(df: pd.DataFrame, cutoff, eval_end) -> bool:
    """Cutoff-eval results, replacing rows for this (cutoff, eval_end)."""
    try:
        c = cutoff.isoformat() if hasattr(cutoff, "isoformat") else str(cutoff)
        e = eval_end.isoformat() if hasattr(eval_end, "isoformat") else str(eval_end)
        rec = df.copy()
        rec["cutoff"] = c
        rec["eval_end"] = e
        rec = _align(rec, _CUTOFF_COLS)
        con = get_conn()
        ensure_schema(con)
        con.execute("DELETE FROM cutoff_eval WHERE cutoff=? AND eval_end=?", (c, e))
        rec.to_sql("cutoff_eval", con, if_exists="append", index=False)
        con.commit(); con.close()
        return True
    except Exception as e:
        print(f"[db_io] WARN write_cutoff_eval failed: {e}")
        return False


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _replace_table(df: pd.DataFrame, table: str, cols: list[str]) -> bool:
    try:
        rec = _align(df, cols)
        con = get_conn()
        ensure_schema(con)
        con.execute(f"DELETE FROM {table}")
        rec.to_sql(table, con, if_exists="append", index=False)
        con.commit(); con.close()
        return True
    except Exception as e:
        print(f"[db_io] WARN write {table} failed: {e}")
        return False


def _num(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _cell(v):
    """Generic scalar coercion for executemany rows (NaN -> None)."""
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, bool):
        return int(v)
    return v
