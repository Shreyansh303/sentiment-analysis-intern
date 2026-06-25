"""
pick_top.py — turn the daily sentiment panel into a LONG-ONLY pick list, and freeze
the backstory behind every pick into an append-only decision ledger.

What it does, for a given snapshot date (default = the latest in snapshots.csv):
  1. Rank the Nifty 50 by agg_sent, drop low-n names, take the top N.
  2. For each pick, reconstruct the EXACT articles that drove its score and how much
     each one contributed — agg_sent is a transparent weighted average
     (sentiment_core._weighted_mean), so every article's contribution is arithmetic:
        contribution_i = sent_i * eff_w_i / sum(eff_w)   (sums back to agg_sent)
     where eff_w = recency * source_trust * impact * scope_multiplier.
  3. Append picks + their reason cards to decision_ledger.csv (NEVER overwritten —
     unlike snapshots.csv, which is rewritten per day). This is the immutable record
     people read later to judge whether each call was sound. grade_ledger.py fills in
     what actually happened.

This reuses the pipeline's own scoring/weighting (no re-running FinBERT): per-article
raw scores come from scored_history.csv, weights from apply_weights(), headlines/URLs
from articles_fetched.json.

Usage:
    python pick_top.py                       # top 5 for the latest snapshot date
    python pick_top.py --top 3               # pick the best 3
    python pick_top.py --date 2026-06-24     # a specific snapshot date
    python pick_top.py --include-low-n       # don't drop thin-coverage names
    python pick_top.py --markdown            # also write a browsable cards_<date>.md
"""
import argparse
import json
import os
from datetime import date

import numpy as np
import pandas as pd

from sentiment_core import Config, SECTOR_OF, apply_weights
from universe import TICKERS, NIFTY_50

SNAPSHOT_FILE = "snapshots.csv"
LEDGER_FILE = "decision_ledger.csv"
ARTICLES_FILE = "articles_fetched.json"
# Reconstruct from the CSV, not load_scored_history() — the DB copy has drifted out of
# sync with snapshots.csv, and we need the EXACT article set that produced each score so
# the reason card's contributions sum back to the snapshot's agg_sent.
SCORED_HISTORY_FILE = "scored_history.csv"
TOP_CARD_ARTICLES = 6   # how many drivers to print per card (the ledger keeps them all)

LEDGER_COLS = [
    "date", "ticker", "name", "rank", "agg_sent", "n_sent", "low_n",
    "article_id", "source", "headline", "url", "art_date",
    "sent", "eff_weight", "weight_share", "contribution",
]


def load_latest_snapshot(snap_date=None, path=SNAPSHOT_FILE):
    """Return (snap_date_iso, rows_df) for the chosen date — the latest by default."""
    panel = pd.read_csv(path)
    if "low_n" in panel:
        panel["low_n"] = panel["low_n"].map(
            lambda v: str(v).strip().lower() in ("true", "1"))
    if snap_date is None:
        snap_date = sorted(panel["date"].unique())[-1]
    rows = panel[panel["date"] == snap_date].copy()
    if rows.empty:
        raise SystemExit(f"No snapshot rows for date {snap_date} in {path}.")
    return snap_date, rows


def load_headlines(path=ARTICLES_FILE):
    """{article_id -> {headline, url, source}} from the live news file. Older snapshot
    dates may predate the current file, so headlines can be missing — that's fine."""
    if not os.path.exists(path):
        return {}
    arts = json.load(open(path))
    out = {}
    for a in arts:
        out[a["id"]] = {
            "headline": str(a.get("text", ""))[:160],
            "url": a.get("url", ""),
            "source": a.get("source", ""),
        }
    return out


def contributing_articles(df_w, tk, cfg):
    """The articles that actually moved tk's agg_sent, with each one's contribution.

    Mirrors sentiment_core._weighted_inputs (same cut-filter + scope multiplier) but
    keeps article identity so we can attribute the score. Returns a frame sorted by
    signed contribution (top positive drivers first); sum(contribution) == agg_sent."""
    sub = df_w[~df_w["cut_sent"]].copy()
    scope = sub["scope"].values
    mult = np.zeros(len(sub))
    mult[scope == tk] = 1.0
    sec = SECTOR_OF.get(tk)
    if sec is not None:
        mult[scope == sec] = cfg.sent_sector_damp
    mult[scope == "MACRO"] = cfg.sent_macro_damp
    sub["mult"] = mult
    sub = sub[sub["mult"] > 0.0].copy()
    sub["eff_weight"] = sub["w_sent"] * sub["mult"]
    sub = sub[sub["eff_weight"] > 0.0].copy()
    tot = sub["eff_weight"].sum()
    if tot <= 0 or sub.empty:
        return sub.iloc[0:0]
    sub["weight_share"] = sub["eff_weight"] / tot          # share of influence (sums to 1)
    sub["contribution"] = sub["sent"] * sub["weight_share"]  # signed points (sums to agg_sent)
    return sub.sort_values("contribution", ascending=False)


def build_cards(snap_date, rows, top, include_low_n):
    """Rank, select top-N, and reconstruct each pick's reason card. Returns
    (picks_df, ledger_rows_df)."""
    cfg = Config()
    today = date.fromisoformat(snap_date)

    # 1. Rank long-only by agg_sent; drop thin-coverage names unless asked to keep.
    pool = rows if include_low_n else rows[~rows["low_n"]]
    pool = pool.sort_values("agg_sent", ascending=False).head(top).reset_index(drop=True)

    # 2. Reconstruct the weighted articles for this snapshot date (no model re-run).
    hist = pd.read_csv(SCORED_HISTORY_FILE)
    hist = hist[hist["snap_date"] == snap_date].copy()
    if hist.empty:
        raise SystemExit(
            f"No scored_history rows for {snap_date}; cannot reconstruct backstory. "
            "Reason cards need the per-article scores from that day.")
    df_w = apply_weights(hist, cfg, today)
    headlines = load_headlines()

    ledger_rows = []
    cards = []
    for rank, r in enumerate(pool.itertuples(index=False), start=1):
        tk = r.ticker
        arts = contributing_articles(df_w, tk, cfg)
        # enrich with human-readable headline/source for the card display
        arts["headline"] = arts["id"].map(lambda i: headlines.get(i, {}).get("headline", ""))
        arts["source"] = arts.apply(
            lambda a: headlines.get(a["id"], {}).get("source") or a["source"], axis=1)
        for a in arts.itertuples(index=False):
            meta = headlines.get(a.id, {})
            ledger_rows.append({
                "date": snap_date, "ticker": tk, "name": NIFTY_50.get(tk, tk),
                "rank": rank, "agg_sent": round(float(r.agg_sent), 4),
                "n_sent": int(r.n_sent), "low_n": bool(r.low_n),
                "article_id": a.id, "source": meta.get("source") or a.source,
                "headline": meta.get("headline", ""), "url": meta.get("url", ""),
                "art_date": str(a.date)[:10],
                "sent": round(float(a.sent), 4),
                "eff_weight": round(float(a.eff_weight), 4),
                "weight_share": round(float(a.weight_share), 4),
                "contribution": round(float(a.contribution), 4),
            })
        cards.append((rank, tk, r, arts))
    return cards, pd.DataFrame(ledger_rows, columns=LEDGER_COLS)


def print_cards(snap_date, cards):
    print(f"\n=== LONG PICKS for {snap_date}  (rank by agg_sent, top {len(cards)}) ===")
    print("Each pick shows WHY: the articles that drove the score and their share of it.")
    for rank, tk, r, arts in cards:
        name = NIFTY_50.get(tk, tk)
        recon = float(arts["contribution"].sum()) if not arts.empty else 0.0
        flag = " (!) LOW-N" if bool(r.low_n) else ""
        print(f"\n#{rank}  {tk:<13} {name}")
        print(f"     score agg_sent = {float(r.agg_sent):+.3f}  "
              f"(n={int(r.n_sent)} articles, reconstructed {recon:+.3f}){flag}")
        if arts.empty:
            print("     └─ no contributing articles found in scored_history")
            continue
        for a in arts.head(TOP_CARD_ARTICLES).itertuples(index=False):
            head = a.headline if isinstance(a.headline, str) and a.headline else "(headline not retained)"
            print(f"     • {a.weight_share*100:4.0f}% | sent={a.sent:+.2f} | "
                  f"{str(a.source)[:22]:<22} {str(a.date)[:10]} | {head[:70]}")
        extra = len(arts) - TOP_CARD_ARTICLES
        if extra > 0:
            print(f"     … +{extra} more contributing articles (all saved to the ledger)")


def append_ledger(ledger_df, path=LEDGER_FILE):
    """Append-only, idempotent per (date, ticker): existing picks are never rewritten,
    so the recorded backstory is immutable once logged."""
    if ledger_df.empty:
        print("Nothing to log.")
        return
    if os.path.exists(path):
        prev = pd.read_csv(path)
        existing = set(zip(prev["date"].astype(str), prev["ticker"].astype(str)))
        mask = [
            (d, t) not in existing
            for d, t in zip(ledger_df["date"].astype(str), ledger_df["ticker"].astype(str))
        ]
        fresh = ledger_df[mask]
        if fresh.empty:
            print(f"All picks for these dates already in {path} (immutable — not rewritten).")
            return
        out = pd.concat([prev, fresh], ignore_index=True)
    else:
        fresh = ledger_df
        out = ledger_df
    out.to_csv(path, index=False)
    n_picks = fresh[["date", "ticker"]].drop_duplicates().shape[0]
    print(f"\nLogged {n_picks} pick(s) / {len(fresh)} article rows to {path} (append-only).")


def write_markdown(snap_date, cards, path=None):
    path = path or f"decision_cards_{snap_date}.md"
    lines = [f"# Long picks — {snap_date}\n",
             "Ranked by FinBERT sentiment (`agg_sent`). Each pick lists the articles "
             "that drove its score and their share of the weighted average.\n"]
    for rank, tk, r, arts in cards:
        lines.append(f"\n## #{rank} {tk} — {NIFTY_50.get(tk, tk)}")
        lines.append(f"\n**Score:** `agg_sent = {float(r.agg_sent):+.3f}` "
                     f"(n={int(r.n_sent)} articles){' — LOW-N ⚠️' if bool(r.low_n) else ''}\n")
        lines.append("| share | sent | source | date | headline |")
        lines.append("|------:|-----:|--------|------|----------|")
        for a in arts.head(TOP_CARD_ARTICLES).itertuples(index=False):
            head = (a.headline or "(headline not retained)").replace("|", "/")
            lines.append(f"| {a.weight_share*100:.0f}% | {a.sent:+.2f} | "
                         f"{str(a.source)[:28]} | {str(a.date)[:10]} | {head[:80]} |")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"Browsable cards written to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5, help="how many stocks to pick")
    ap.add_argument("--date", default=None, help="snapshot date (YYYY-MM-DD); default latest")
    ap.add_argument("--include-low-n", action="store_true",
                    help="keep names flagged low_n (thin article coverage)")
    ap.add_argument("--no-log", action="store_true", help="print only; don't touch the ledger")
    ap.add_argument("--markdown", action="store_true", help="also write a browsable .md")
    ap.add_argument("--snapshots", default=SNAPSHOT_FILE)
    args = ap.parse_args()

    snap_date, rows = load_latest_snapshot(args.date, args.snapshots)
    cards, ledger_df = build_cards(snap_date, rows, args.top, args.include_low_n)
    print_cards(snap_date, cards)
    if not args.no_log:
        append_ledger(ledger_df)
    if args.markdown:
        write_markdown(snap_date, cards)
    print("\nReminder: a high sentiment score is NOT a guarantee. Run grade_ledger.py "
          "after the horizon passes to see whether these calls were actually right.")


if __name__ == "__main__":
    main()
