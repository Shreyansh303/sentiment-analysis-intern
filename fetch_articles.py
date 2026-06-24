"""
fetch_articles.py — automated acquisition layer for the sentiment/news pipeline.
Sources: yfinance (Yahoo Finance news) + Google News RSS (Indian coverage).
Output: articles JSON in the exact schema pipeline_nifty.py consumes.

Run locally:  pip install yfinance feedparser
              python fetch_articles.py
Optional full-text extraction: pip install trafilatura
"""
import json, re, hashlib, time
from datetime import datetime, timezone, timedelta

import yfinance as yf
import feedparser

# ---------------- config ----------------
from universe import NIFTY_50   # full Nifty 50, consistent with run_universe_all
# One Google News query per name; yfinance is also queried per ticker symbol below.
TICKERS = {tk: [f"{name} share NSE"] for tk, name in NIFTY_50.items()}
SECTOR_QUERIES = {"SECTOR_IT": ["Nifty IT index", "Indian IT stocks"]}
MACRO_QUERIES  = {"MACRO": ["Nifty Sensex RBI", "FII flows Indian markets", "Fed rate decision"]}
MAX_AGE_DAYS   = 14          # hard recency cutoff at fetch time
PER_QUERY_CAP  = 10
OUTFILE        = "articles_fetched.json"

# ---------------- helpers ----------------
def norm_title(t):
    """Normalize title for dedup: lowercase, strip punctuation/whitespace."""
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

def make_id(title):
    return hashlib.md5(norm_title(title).encode()).hexdigest()[:10]

def too_old(dt):
    return (datetime.now(timezone.utc) - dt) > timedelta(days=MAX_AGE_DAYS)

# ---------------- source 1: yfinance ----------------
def fetch_yfinance(ticker):
    """Yahoo Finance news. Handles both pre- and post-2024 item formats."""
    out = []
    try:
        items = yf.Ticker(ticker).get_news() or []
    except Exception as e:
        print(f"[yf] {ticker} failed: {e}")
        return out
    for it in items:
        c = it.get("content", it)            # new format nests under 'content'
        title = c.get("title", "")
        summary = c.get("summary") or c.get("description") or ""
        # publish time: new format ISO string 'pubDate'; old format epoch
        if "pubDate" in c:
            dt = datetime.fromisoformat(c["pubDate"].replace("Z", "+00:00"))
        elif "providerPublishTime" in it:
            dt = datetime.fromtimestamp(it["providerPublishTime"], tz=timezone.utc)
        else:
            continue
        if too_old(dt) or not title:
            continue
        provider = (c.get("provider") or {}).get("displayName", "yahoo")
        url = (c.get("canonicalUrl") or {}).get("url", it.get("link", ""))
        out.append(dict(id=make_id(title), scope=ticker,
                        source=f"{provider} (via yfinance)",
                        date=dt.date().isoformat(),
                        text=f"{title}. {summary}".strip(), url=url))
    return out

# ---------------- source 2: Google News RSS ----------------
def fetch_gnews(query, scope):
    """Free, no key. Good for ET/Moneycontrol/BS coverage of Indian names."""
    url = ("https://news.google.com/rss/search?"
           f"q={query.replace(' ', '+')}+when:{MAX_AGE_DAYS}d&hl=en-IN&gl=IN&ceid=IN:en")
    out = []
    feed = feedparser.parse(url)
    for e in feed.entries[:PER_QUERY_CAP]:
        try:
            dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            continue
        if too_old(dt):
            continue
        title = e.get("title", "")
        # Google News titles end with " - Publisher"
        source = title.rsplit(" - ", 1)[-1] if " - " in title else "google-news"
        summary = re.sub(r"<[^>]+>", "", e.get("summary", ""))[:400]
        out.append(dict(id=make_id(title), scope=scope,
                        source=f"{source} (via gnews-rss)",
                        date=dt.date().isoformat(),
                        text=f"{title}. {summary}".strip(),
                        url=e.get("link", "")))
    return out

# ---------------- optional: full article text ----------------
def hydrate_full_text(article):
    """Title+summary is often enough for the LLM analysts; pull full body
    only for HIGH-impact candidates to save bandwidth/tokens."""
    try:
        # pyrefly: ignore [missing-import]
        import trafilatura
        html = trafilatura.fetch_url(article["url"])
        body = trafilatura.extract(html) or ""
        if len(body) > 200:
            article["text"] = body[:3000]    # cap tokens per article
    except Exception:
        pass
    return article

# ---------------- main ----------------
def main():
    pool, seen = [], set()

    def add(items):
        for a in items:
            if a["id"] in seen:              # dedup: same story, many outlets
                continue
            seen.add(a["id"])
            pool.append(a)

    for tk, queries in TICKERS.items():
        add(fetch_yfinance(tk))
        for q in queries:
            add(fetch_gnews(q, tk)); time.sleep(1)   # be polite, avoid blocks
    for scope, queries in {**SECTOR_QUERIES, **MACRO_QUERIES}.items():
        for q in queries:
            add(fetch_gnews(q, scope)); time.sleep(1)

    pool.sort(key=lambda a: a["date"], reverse=True)
    json.dump(pool, open(OUTFILE, "w"), indent=1)
    # Dual-write: also upsert into the database (articles table, batch="live").
    try:
        import db_io
        if db_io.upsert_articles(pool, source_batch="live"):
            print(f"  (DB) upserted {len(pool)} articles into articles table.")
    except Exception as e:
        print(f"  (DB) articles write skipped: {e}")
    print(f"fetched {len(pool)} unique articles -> {OUTFILE}")
    by_scope = {}
    for a in pool:
        by_scope[a["scope"]] = by_scope.get(a["scope"], 0) + 1
    for s, n in sorted(by_scope.items()):
        print(f"  {s:<13} {n}")

if __name__ == "__main__":
    main()
