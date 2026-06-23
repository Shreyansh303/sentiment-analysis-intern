"""
fetch_universe.py — fetch a month of news for the whole Nifty-50 universe.

One Google News RSS query per ticker (absolute after:/before: date range) plus
shared MACRO queries, into the standard article schema. Writes one JSON per month
(articles_uni_<mon>.json). Idempotent: skips a month whose file already exists.

Run:  python fetch_universe.py            # all months in universe.MONTH_FETCH
      python fetch_universe.py --month may
"""
import argparse, json, os, re, hashlib, time
from datetime import datetime, timezone, date

import feedparser

from universe import NIFTY_50, MACRO_QUERIES, MONTH_FETCH

PER_QUERY_CAP = 15


def norm_title(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

def make_id(title):
    return hashlib.md5(norm_title(title).encode()).hexdigest()[:10]


def gnews(query, scope, start: date, end: date):
    before_excl = date.fromordinal(end.toordinal() + 1)
    q = f"{query} after:{start.isoformat()} before:{before_excl.isoformat()}"
    url = ("https://news.google.com/rss/search?"
           f"q={q.replace(' ', '+').replace(':', '%3A')}&hl=en-IN&gl=IN&ceid=IN:en")
    out = []
    for e in feedparser.parse(url).entries[:PER_QUERY_CAP]:
        try:
            dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            continue
        if not (start <= dt.date() <= end):
            continue
        title = e.get("title", "")
        if not title:
            continue
        source = title.rsplit(" - ", 1)[-1] if " - " in title else "google-news"
        summary = re.sub(r"<[^>]+>", "", e.get("summary", ""))[:400]
        out.append(dict(id=make_id(title), scope=scope,
                        source=f"{source} (via gnews-rss)",
                        date=dt.date().isoformat(),
                        text=f"{title}. {summary}".strip(), url=e.get("link", "")))
    return out


def fetch_month(mon, start_s, end_s):
    outfile = f"articles_uni_{mon}.json"
    if os.path.exists(outfile):
        print(f"[{mon}] {outfile} exists -> skip")
        return
    start, end = date.fromisoformat(start_s), date.fromisoformat(end_s)
    pool, seen = [], set()

    def add(items):
        for a in items:
            if a["id"] in seen:
                continue
            seen.add(a["id"]); pool.append(a)

    queries = [(name + " share NSE", tk) for tk, name in NIFTY_50.items()]
    queries += [(q, "MACRO") for q in MACRO_QUERIES]

    for i, (q, scope) in enumerate(queries, 1):
        try:
            add(gnews(q, scope, start, end))
        except Exception as ex:
            print(f"[{mon}] query failed ({scope}): {ex}")
        if i % 10 == 0:
            print(f"[{mon}] {i}/{len(queries)} queries, {len(pool)} articles so far")
        time.sleep(0.7)

    pool.sort(key=lambda a: a["date"], reverse=True)
    json.dump(pool, open(outfile, "w"), indent=1)
    # Dual-write: also upsert into the database (articles table, batch="uni_<mon>").
    try:
        import db_io
        if db_io.upsert_articles(pool, source_batch=f"uni_{mon}"):
            print(f"[{mon}] (DB) upserted {len(pool)} articles into articles table.")
    except Exception as e:
        print(f"[{mon}] (DB) articles write skipped: {e}")
    n_scope = len({a["scope"] for a in pool})
    print(f"[{mon}] DONE: {len(pool)} articles across {n_scope} scopes -> {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default=None, help="one of jan/feb/.../jun; default = all")
    args = ap.parse_args()
    for mon, s, e in MONTH_FETCH:
        if args.month and mon != args.month:
            continue
        fetch_month(mon, s, e)


if __name__ == "__main__":
    main()
