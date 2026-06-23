#!/usr/bin/env python3
"""
check_setup.py — pre-demo green-light check.

Verifies, without running anything expensive, that this machine can run the
sentiment-model demo: Python deps installed, the database present + populated,
and the key scripts/data files in place. Stdlib-only, so it always runs.

Usage:  python check_setup.py
"""
import importlib.util
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "db", "sentiment.db")

OK, BAD, WARN = "  [ OK ]", "  [FAIL]", "  [WARN]"
problems = []   # things that would block the demo
warnings = []   # things worth knowing but non-blocking


def line(status, msg):
    print(f"{status}  {msg}")


# --------------------------------------------------------------------------- #
print("\n" + "=" * 64)
print("  PRE-DEMO SETUP CHECK")
print("=" * 64)

# 1) Python version -------------------------------------------------------- #
print("\nPython")
v = sys.version_info
if v >= (3, 8):
    line(OK, f"Python {v.major}.{v.minor}.{v.micro}")
else:
    line(BAD, f"Python {v.major}.{v.minor} — need 3.8+")
    problems.append("Python too old")

# 2) Dependencies ---------------------------------------------------------- #
print("\nDependencies (needed for the LIVE scripts: pipeline_nifty / evaluate_cutoff / backtest)")
deps = {
    "numpy": "live + analysis",
    "pandas": "live + analysis",
    "scipy": "backtest stats (IC, t-tests)",
    "transformers": "FinBERT model",
    "torch": "FinBERT backend",
    "yfinance": "price fetch",
    "feedparser": "news fetch",
}
for mod, why in deps.items():
    if importlib.util.find_spec(mod) is not None:
        line(OK, f"{mod:<14} ({why})")
    else:
        line(BAD, f"{mod:<14} MISSING — {why}")
        problems.append(f"pip install {mod}")

# 3) Database -------------------------------------------------------------- #
print("\nDatabase  (db/sentiment.db)")
# Tables that must exist and be non-empty. Counts grow as the pipeline re-runs,
# so we report the live count rather than comparing to a fixed expected value.
tables = ["articles", "article_scores", "snapshots", "universe_calls", "backtest_pnl"]
if not os.path.exists(DB):
    line(BAD, "sentiment.db not found — run: cd db && python3 2026-06-22-migrate.py")
    problems.append("database missing")
else:
    line(OK, f"file present ({os.path.getsize(DB)/1024:.0f} KB)")
    try:
        con = sqlite3.connect(DB)
        con.execute("PRAGMA foreign_keys = ON;")
        for tbl in tables:
            try:
                n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            except sqlite3.Error as e:
                line(BAD, f"{tbl:<16} table error: {e}")
                problems.append(f"table {tbl} unreadable")
                continue
            if n == 0:
                line(BAD, f"{tbl:<16} 0 rows — empty")
                problems.append(f"table {tbl} empty")
            else:
                line(OK, f"{tbl:<16} {n} rows")

        # integrity + foreign keys
        fk = con.execute("PRAGMA foreign_key_check;").fetchall()
        line(OK, "foreign_key_check clean") if not fk else (
            line(BAD, f"foreign_key_check: {len(fk)} violation(s)"),
            problems.append("FK violations"))
        integ = con.execute("PRAGMA integrity_check;").fetchone()[0]
        line(OK, "integrity_check ok") if integ == "ok" else (
            line(BAD, f"integrity_check: {integ}"),
            problems.append("integrity check failed"))
        con.close()
    except sqlite3.Error as e:
        line(BAD, f"could not open DB: {e}")
        problems.append("DB open failed")

# 4) Key files ------------------------------------------------------------- #
print("\nKey scripts & data files")
files = [
    "pipeline_nifty.py", "backtest.py", "evaluate_cutoff.py", "sensitivity.py",
    "run_universe_all.py", "run_combined_book.py", "sentiment_core.py", "db_io.py",
    "articles_fetched.json", "snapshots.csv",
    "universe_calls.csv", "combined_book_pnl.csv",
    os.path.join("db", "2026-06-22-README.md"),
]
for f in files:
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        line(OK, f)
    else:
        line(BAD, f"{f} MISSING")
        problems.append(f"missing file {f}")

# 5) sqlite3 CLI (optional convenience) ------------------------------------ #
print("\nOptional")
from shutil import which
if which("sqlite3"):
    line(OK, "sqlite3 CLI available (interactive queries will work)")
else:
    line(WARN, "sqlite3 CLI not found — use the python3 -c one-liner from the runbook instead")
    warnings.append("no sqlite3 CLI")

# --------------------------------------------------------------------------- #
print("\n" + "=" * 64)
if not problems:
    print("  GREEN LIGHT — everything needed for the demo is in place.")
    if warnings:
        print(f"  ({len(warnings)} non-blocking note(s) above.)")
    print("=" * 64 + "\n")
    sys.exit(0)
else:
    print(f"  NOT READY — {len(problems)} blocking issue(s):")
    for p in problems:
        print(f"     - {p}")
    print("\n  Fix tip: pip install transformers torch yfinance scipy feedparser numpy pandas")
    print("=" * 64 + "\n")
    sys.exit(1)
