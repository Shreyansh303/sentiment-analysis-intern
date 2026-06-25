"""
One-shot helper: export 2026 monthly articles from the DB back to
articles_uni_<mon>.json so run_universe_all.py can score and backtest them.

Usage:
    python export_jan_from_db.py            # all months jan..may (skips jun)
    python export_jan_from_db.py jan mar    # only the named months
"""
import sys
import json
import sqlite3

DB_PATH = "db/sentiment.db"

# month -> (first day, last day) inclusive. June is the live month and is
# fetched fresh elsewhere, so it is excluded from the historical export.
MONTHS = {
    "jan": ("2026-01-01", "2026-01-31"),
    "feb": ("2026-02-01", "2026-02-28"),
    "mar": ("2026-03-01", "2026-03-31"),
    "apr": ("2026-04-01", "2026-04-30"),
    "may": ("2026-05-01", "2026-05-31"),
}

cols = ["id", "scope", "source", "date", "text", "url"]


def export_month(con, mon):
    start, end = MONTHS[mon]
    rows = con.execute(
        """SELECT id, ticker AS scope, source, published_date AS date, text, url
           FROM articles
           WHERE published_date BETWEEN ? AND ?
           ORDER BY published_date DESC""",
        (start, end),
    ).fetchall()
    articles = [dict(zip(cols, r)) for r in rows]
    out = f"articles_uni_{mon}.json"
    json.dump(articles, open(out, "w"), indent=1)
    n_tk = len({a["scope"] for a in articles})
    print(f"[{mon}] exported {len(articles)} articles across {n_tk} scopes -> {out}")


def main():
    wanted = sys.argv[1:] or list(MONTHS)
    bad = [m for m in wanted if m not in MONTHS]
    if bad:
        sys.exit(f"unknown month(s): {bad}; choose from {list(MONTHS)}")
    con = sqlite3.connect(DB_PATH)
    for mon in wanted:
        export_month(con, mon)
    con.close()


if __name__ == "__main__":
    main()
