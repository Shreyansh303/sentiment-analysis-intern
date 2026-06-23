-- Sentiment-analysis pipeline schema
-- Written for PostgreSQL; runs on SQLite with the noted substitutions.
-- 2026-06-22
--
-- Portability notes (SQLite):
--   * SQLite ignores most type lengths and uses dynamic typing, but the
--     declarations below are valid in both engines.
--   * For Postgres you may swap TEXT primary keys for the same TEXT type;
--     no change needed. NUMERIC works in both.
--   * CHECK / FOREIGN KEY constraints work in both (enable in SQLite with
--     `PRAGMA foreign_keys = ON;` — the loader does this).

-- ---------------------------------------------------------------------------
-- articles: one row per fetched news article (deduplicated on content id)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS articles (
    id              TEXT        PRIMARY KEY,          -- content hash from pipeline
    ticker          TEXT        NOT NULL,             -- was "scope" in source JSON
    source          TEXT        NOT NULL,
    published_date  DATE        NOT NULL,             -- was "date" in source JSON
    url             TEXT,
    text            TEXT        NOT NULL
);

-- ---------------------------------------------------------------------------
-- article_scores: per-article sentiment scoring (scored_history.csv)
-- One score row references exactly one article via id.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS article_scores (
    id          TEXT        NOT NULL,                 -- FK -> articles.id
    snap_date   DATE        NOT NULL,                 -- date the scoring snapshot was taken
    scope       TEXT        NOT NULL,                 -- ticker (kept for convenience)
    source      TEXT,
    article_date DATE,                                -- article publish date at scoring time
    pos_prob    NUMERIC     CHECK (pos_prob  IS NULL OR (pos_prob  BETWEEN 0 AND 1)),
    neg_prob    NUMERIC     CHECK (neg_prob  IS NULL OR (neg_prob  BETWEEN 0 AND 1)),
    neu_prob    NUMERIC     CHECK (neu_prob  IS NULL OR (neu_prob  BETWEEN 0 AND 1)),
    sent        NUMERIC     CHECK (sent IS NULL OR (sent BETWEEN -1 AND 1)),
    news        NUMERIC,
    pos_count   INTEGER,
    neg_count   INTEGER,
    PRIMARY KEY (id, snap_date),
    FOREIGN KEY (id) REFERENCES articles (id)
);

-- ---------------------------------------------------------------------------
-- snapshots: live daily aggregate signal per ticker (snapshots.csv)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS snapshots (
    date        DATE        NOT NULL,
    ticker      TEXT        NOT NULL,
    agg_sent    NUMERIC,
    agg_news    NUMERIC,
    divergence  NUMERIC,
    label       TEXT,
    n_sent      INTEGER,
    n_news      INTEGER,
    sent_lo     NUMERIC,                              -- NULL = confidence bound not computed
    sent_hi     NUMERIC,
    news_lo     NUMERIC,
    news_hi     NUMERIC,
    low_n       BOOLEAN,
    PRIMARY KEY (date, ticker)
);

-- ---------------------------------------------------------------------------
-- universe_calls: backtest universe with realized returns + residuals
-- (universe_calls.csv) -- superset of snapshots columns
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS universe_calls (
    date        DATE        NOT NULL,
    ticker      TEXT        NOT NULL,
    agg_sent    NUMERIC,
    agg_news    NUMERIC,
    divergence  NUMERIC,
    label       TEXT,
    n_sent      INTEGER,
    n_news      INTEGER,
    sent_lo     NUMERIC,
    sent_hi     NUMERIC,
    news_lo     NUMERIC,
    news_hi     NUMERIC,
    low_n       BOOLEAN,
    combined    NUMERIC,
    actual_ret  NUMERIC,
    momentum    NUMERIC,
    window      TEXT,
    ret_resid   NUMERIC,
    sig_resid   NUMERIC,
    PRIMARY KEY (date, ticker)
);

-- ---------------------------------------------------------------------------
-- backtest_pnl: per-window book returns (combined_book_pnl.csv)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_pnl (
    window      TEXT        PRIMARY KEY,
    ret         NUMERIC,
    turnover    NUMERIC
);

-- ---------------------------------------------------------------------------
-- Indexes for the common (ticker, date) access pattern
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_articles_ticker_date   ON articles (ticker, published_date);
CREATE INDEX IF NOT EXISTS idx_scores_scope_snap      ON article_scores (scope, snap_date);
CREATE INDEX IF NOT EXISTS idx_snapshots_ticker       ON snapshots (ticker, date);
CREATE INDEX IF NOT EXISTS idx_universe_ticker        ON universe_calls (ticker, date);
