-- Schema v2 — additions for the script migration (2026-06-22)
-- Adds tables for secondary pipeline outputs + a source_batch tag on articles.
-- Idempotent: safe to run multiple times. db_io.py also applies these
-- automatically on first use, so running this by hand is optional.

-- Tag which fetch batch an article came from: "live" (fetch_articles.py) or
-- "uni_<mon>" (fetch_universe.py). Existing rows stay NULL.
-- (SQLite has no "ADD COLUMN IF NOT EXISTS"; db_io applies this in a try/except.
--  If running by hand and the column already exists, ignore the error.)
ALTER TABLE articles ADD COLUMN source_batch TEXT;

-- Per-window cross-sectional metrics (run_universe_all.py -> universe_windows.csv)
CREATE TABLE IF NOT EXISTS universe_windows (
    window      TEXT    PRIMARY KEY,
    n           INTEGER,
    ic_sent     NUMERIC,
    ic_mom      NUMERIC,
    ls_sent     NUMERIC,
    ls_mom      NUMERIC
);

-- Per-run backtest P&L curves (backtest.py -> pnl_<signal>_<horizon>.csv)
CREATE TABLE IF NOT EXISTS backtest_pnl_runs (
    signal      TEXT    NOT NULL,
    horizon     TEXT    NOT NULL,
    date        DATE    NOT NULL,
    ret         NUMERIC,
    cum         NUMERIC,
    PRIMARY KEY (signal, horizon, date)
);

-- Point-in-time cutoff evaluation results (evaluate_cutoff.py -> cutoff_eval_results.csv)
CREATE TABLE IF NOT EXISTS cutoff_eval (
    cutoff      DATE    NOT NULL,
    eval_end    DATE    NOT NULL,
    ticker      TEXT    NOT NULL,
    agg_sent    NUMERIC,
    agg_news    NUMERIC,
    combined    NUMERIC,
    label       TEXT,
    prediction  TEXT,
    n_sent      INTEGER,
    n_news      INTEGER,
    low_n       BOOLEAN,
    entry_date  DATE,
    exit_date   DATE,
    entry_px    NUMERIC,
    exit_px     NUMERIC,
    actual_ret  NUMERIC,
    actual_dir  TEXT,
    PRIMARY KEY (cutoff, eval_end, ticker)
);
