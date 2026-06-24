"""
fetch_earnings.py — fetch and score quarterly earnings call transcripts for
Nifty 50 Indian companies. Produces scored_earnings.csv (one row per ticker per
quarter), which run_universe_all.py reads to attach agg_earnings to each window.

Design mirrors the "score once / reweight many" principle of the news pipeline:
  - Transcripts are fetched once per quarter and cached.
  - Scoring is idempotent: skips tickers already present in the cache.
  - get_earnings_for_window() is the read-path used by run_universe_all.py.

Run:
    python fetch_earnings.py --quarter Q3FY26
    python fetch_earnings.py --quarter Q4FY26
    python fetch_earnings.py --quarter all
    python fetch_earnings.py --quarter Q3FY26 --tickers RELIANCE.NS,TCS.NS,INFY.NS
"""
from __future__ import annotations

import argparse
import io
import os
import time
from datetime import date, datetime
from urllib.parse import quote

import requests
import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

from universe import NIFTY_50
from sentiment_core import Config, load_models, score_transcript

SCORED_EARNINGS_FILE = "scored_earnings.csv"
MIN_TRANSCRIPT_LENGTH = 1500   # chars; shorter = press release summary, not transcript
RATE_SLEEP = 1.5               # seconds between network calls

QUARTER_WINDOWS = {
    # Indian fiscal year is Apr->Mar. Each concall lands ~1 month after quarter end.
    "Q1FY25": {"quarter_end": "2024-06-30", "after": "2024-07-01", "before": "2024-08-31", "label": "Q1"},
    "Q2FY25": {"quarter_end": "2024-09-30", "after": "2024-10-01", "before": "2024-11-30", "label": "Q2"},
    "Q3FY25": {"quarter_end": "2024-12-31", "after": "2025-01-01", "before": "2025-02-28", "label": "Q3"},
    "Q4FY25": {"quarter_end": "2025-03-31", "after": "2025-04-01", "before": "2025-05-31", "label": "Q4"},
    "Q1FY26": {"quarter_end": "2025-06-30", "after": "2025-07-01", "before": "2025-08-31", "label": "Q1"},
    "Q2FY26": {"quarter_end": "2025-09-30", "after": "2025-10-01", "before": "2025-11-30", "label": "Q2"},
    "Q3FY26": {"quarter_end": "2025-12-31", "after": "2026-01-01", "before": "2026-02-28", "label": "Q3"},
    "Q4FY26": {"quarter_end": "2026-03-31", "after": "2026-04-01", "before": "2026-05-31", "label": "Q4"},
}

_SCORED_COLS = ["ticker", "quarter", "quarter_end", "transcript_date",
                "finbert_sent", "lm_polarity", "lm_uncertainty", "lm_modal",
                "n_chunks", "source", "agg_earnings"]


# --------------------------------------------------------------------------- #
# Fetch layer  (screener.in concall transcripts)
#
# Screener lists every quarterly earnings ("concall") on the company page inside a
# <div class="documents concalls"> block: one <li> per call, holding the call month
# (e.g. "Apr 2026") and an <a class="concall-link">Transcript</a> link to the raw
# transcript PDF (hosted on BSE). We pick the concall whose month falls inside the
# quarter's reporting window, download that PDF, and extract its text.
# --------------------------------------------------------------------------- #

SCREENER_BASE = "https://www.screener.in/company"
HTTP_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
}
HTTP_TIMEOUT = 40


def _screener_code(ticker: str) -> str:
    """Nifty yfinance ticker -> screener.in company code (the NSE symbol)."""
    return ticker.replace(".NS", "")


def _fetch_concall_page(code: str) -> str:
    """Return the screener company-page HTML (consolidated first, then standalone)."""
    for suffix in ("consolidated/", ""):
        url = f"{SCREENER_BASE}/{quote(code)}/{suffix}"
        try:
            r = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        except Exception:
            continue
        if r.status_code == 200 and "concall" in r.text.lower():
            return r.text
    return ""


def _list_concalls(html: str) -> list[dict]:
    """Parse the Concalls block -> [{month, url}] for rows that have a transcript link."""
    soup = BeautifulSoup(html, "lxml")
    block = soup.find("div", class_="concalls")
    if block is None:
        return []
    calls = []
    for li in block.select("ul.list-links li"):
        label = li.find("div")
        link = next((a for a in li.select("a.concall-link")
                     if a.get_text(strip=True) == "Transcript" and a.get("href")), None)
        if label is None or link is None:
            continue
        try:
            month = datetime.strptime(label.get_text(strip=True), "%b %Y").date()
        except ValueError:
            continue
        calls.append({"month": month, "url": link["href"]})
    return calls


def _extract_pdf_text(url: str) -> str:
    """Download a transcript PDF and extract its text. Returns "" on failure."""
    try:
        r = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if r.status_code != 200 or "pdf" not in r.headers.get("content-type", "").lower():
            return ""
        reader = PdfReader(io.BytesIO(r.content))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""


def fetch_transcript_text(ticker: str, company_name: str, quarter: str) -> dict | None:
    """Fetch an earnings-call transcript for (ticker, quarter) from screener.in.

    Finds the concall whose month falls inside the quarter's reporting window,
    downloads its transcript PDF, and returns {text, source, transcript_date, url}
    when the extracted body is >= MIN_TRANSCRIPT_LENGTH chars, else None.
    """
    qw = QUARTER_WINDOWS[quarter]
    after = date.fromisoformat(qw["after"])
    before = date.fromisoformat(qw["before"])

    html = _fetch_concall_page(_screener_code(ticker))
    time.sleep(RATE_SLEEP)
    if not html:
        return None

    # newest first, so a re-reported quarter prefers the latest transcript
    for call in sorted(_list_concalls(html), key=lambda c: c["month"], reverse=True):
        if not (after <= call["month"] <= before):
            continue
        body = _extract_pdf_text(call["url"])
        time.sleep(RATE_SLEEP)
        if len(body) >= MIN_TRANSCRIPT_LENGTH:
            return {
                "text": body,
                "source": "screener.in (BSE transcript)",
                "transcript_date": call["month"].isoformat(),
                "url": call["url"],
            }
    return None


# --------------------------------------------------------------------------- #
# Scoring and cache
# --------------------------------------------------------------------------- #

def load_earnings_scores(path: str = SCORED_EARNINGS_FILE) -> pd.DataFrame:
    """Read from DB first; fall back to CSV; return empty DataFrame if neither exists."""
    try:
        import db_io
        df = db_io.read_earnings_scores()
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f"[earnings] DB read skipped: {e}")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=_SCORED_COLS)


def _save_earnings_scores(new_df: pd.DataFrame, path: str = SCORED_EARNINGS_FILE) -> None:
    """Upsert new rows into the CSV on (ticker, quarter). Dual-writes to DB."""
    existing = load_earnings_scores(path)
    if not existing.empty:
        # drop any rows that will be replaced
        key = existing.set_index(["ticker", "quarter"]).index
        new_key = new_df.set_index(["ticker", "quarter"]).index
        existing = existing[~existing.set_index(["ticker", "quarter"]).index.isin(new_key)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(path, index=False)
    try:
        import db_io
        db_io.upsert_earnings_scores(new_df)
    except Exception as e:
        print(f"[earnings] DB write skipped: {e}")


def score_and_cache_quarter(quarter: str, models=None,
                            tickers: list[str] | None = None,
                            force: bool = False) -> pd.DataFrame:
    """Fetch + score all tickers for one quarter. Idempotent: skips already-cached tickers."""
    existing = load_earnings_scores()
    if not existing.empty and not force:
        already_done = set(existing.query("quarter == @quarter")["ticker"])
    else:
        already_done = set()

    if models is None:
        models = load_models()

    target = tickers or list(NIFTY_50.keys())
    new_rows = []
    for ticker in target:
        company_name = NIFTY_50.get(ticker, ticker)
        if ticker in already_done:
            print(f"  [{quarter}] {ticker:<20} already cached -> skip")
            continue
        print(f"  [{quarter}] {ticker:<20} fetching transcript...", end=" ", flush=True)
        result = fetch_transcript_text(ticker, company_name, quarter)
        if result is None:
            print("NO transcript found")
            continue
        scores = score_transcript(result["text"], models=models)
        row = {
            "ticker": ticker,
            "quarter": quarter,
            "quarter_end": QUARTER_WINDOWS[quarter]["quarter_end"],
            "transcript_date": result["transcript_date"],
            "source": result["source"],
            **scores,
        }
        new_rows.append(row)
        print(f"agg_earnings={scores['agg_earnings']:+.3f}  n_chunks={scores['n_chunks']}")

    if new_rows:
        _save_earnings_scores(pd.DataFrame(new_rows))
        print(f"  [{quarter}] saved {len(new_rows)} new rows -> {SCORED_EARNINGS_FILE}")
    else:
        print(f"  [{quarter}] no new rows to save (all cached or no transcripts found)")

    return load_earnings_scores()


# --------------------------------------------------------------------------- #
# Read-path: called by run_universe_all.py per window
# --------------------------------------------------------------------------- #

def get_earnings_for_window(cutoff_date: date, cfg: Config) -> dict[str, float]:
    """Return {ticker: agg_earnings_weighted} for all tickers with a non-stale
    transcript as of cutoff_date. Staleness-weighted: score decays toward 0
    at the earnings_half_life. Returns empty dict if no scored transcripts exist."""
    df = load_earnings_scores()
    if df.empty:
        return {}

    df["transcript_date"] = pd.to_datetime(df["transcript_date"]).dt.date
    df = df[df["transcript_date"] <= cutoff_date].copy()
    if df.empty:
        return {}

    df["age_days"] = df["transcript_date"].apply(lambda d: (cutoff_date - d).days)
    df = df[df["age_days"] <= cfg.earnings_max_age]
    if df.empty:
        return {}

    # per ticker: take only the most recent transcript within the window
    df = df.sort_values("transcript_date").groupby("ticker").last().reset_index()

    # staleness decay: score at age=half_life is 0.5× the raw score
    df["stale_w"] = df["age_days"].apply(lambda a: 0.5 ** (a / cfg.earnings_half_life))
    df["agg_earnings_weighted"] = (df["agg_earnings"] * df["stale_w"]).round(4)

    return dict(zip(df["ticker"], df["agg_earnings_weighted"]))


# --------------------------------------------------------------------------- #
# Coverage summary
# --------------------------------------------------------------------------- #

def print_coverage(df: pd.DataFrame) -> None:
    if df.empty:
        print("  No earnings scores found.")
        return
    print(f"\n{'Quarter':<10} {'N tickers':>10} {'Mean agg_earn':>14} "
          f"{'Median agg_earn':>16} {'Sources'}")
    print("-" * 70)
    for q, g in df.groupby("quarter"):
        src_counts = g["source"].str.extract(r"\(via (.+?)\)")[0].value_counts()
        src_str = ", ".join(f"{v}×{k}" for k, v in src_counts.items())
        print(f"  {q:<10} {len(g):>10} {g['agg_earnings'].mean():>14.3f} "
              f"{g['agg_earnings'].median():>16.3f}  {src_str}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Fetch and score earnings call transcripts.")
    parser.add_argument("--quarter", choices=list(QUARTER_WINDOWS.keys()) + ["all"],
                        default="all", help="Which quarter(s) to fetch.")
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated list of tickers (default: all Nifty 50).")
    parser.add_argument("--force", action="store_true",
                        help="Re-score even if already cached.")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
    quarters = list(QUARTER_WINDOWS.keys()) if args.quarter == "all" else [args.quarter]

    print(f"Loading models (FinBERT + LM)...")
    models = load_models()

    for q in quarters:
        print(f"\n=== {q} ===")
        score_and_cache_quarter(q, models=models, tickers=tickers, force=args.force)

    df = load_earnings_scores()
    print(f"\n=== Coverage summary ({len(df)} total rows) ===")
    print_coverage(df)


if __name__ == "__main__":
    main()
