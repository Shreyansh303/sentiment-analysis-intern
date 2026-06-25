"""
grade_ledger.py — score the decision ledger against what actually happened.

pick_top.py freezes WHY each stock was picked. This fills in the OUTCOME: for every
ledgered pick, it looks up the realized forward return over the holding horizon and
marks whether the call was right. It NEVER edits decision_ledger.csv (the reason cards
stay immutable) — it writes a derived view, ledger_graded.csv, that can be re-run as
more sessions become available.

Leakage rule matches the backtest: a pick dated D is entered at the NEXT session's open
and held `horizon` sessions (open-to-open), so the pick day never touches the return.

Usage:
    python grade_ledger.py                 # grade at the 5-session horizon
    python grade_ledger.py --horizon 1     # 1-session horizon
"""
import argparse

import numpy as np
import pandas as pd

from backtest import fetch_prices, forward_return

LEDGER_FILE = "results/decision_ledger.csv"
GRADED_FILE = "results/ledger_graded.csv"


def unique_picks(path=LEDGER_FILE):
    """Collapse the per-article ledger to one row per (date, ticker) pick."""
    led = pd.read_csv(path, parse_dates=["date"])
    led["date"] = led["date"].dt.normalize()
    picks = (led.groupby(["date", "ticker"], as_index=False)
                .agg(name=("name", "first"), rank=("rank", "first"),
                     agg_sent=("agg_sent", "first"), n_sent=("n_sent", "first"),
                     low_n=("low_n", "first"), n_articles=("article_id", "nunique")))
    return picks.sort_values(["date", "rank"]).reset_index(drop=True)


def grade(picks, horizon):
    opens = fetch_prices(picks["ticker"].unique(), picks["date"].min(), picks["date"].max())
    rets, status = [], []
    for _, p in picks.iterrows():
        s = opens.get(p["ticker"])
        if s is None:
            rets.append(np.nan); status.append("no-prices"); continue
        r = forward_return(s, p["date"], horizon)
        rets.append(r)
        status.append("pending" if pd.isna(r) else ("right" if r > 0 else "wrong"))
    out = picks.copy()
    out["horizon"] = horizon
    out["realized_return"] = rets
    out["was_right"] = [None if st in ("pending", "no-prices") else (st == "right")
                        for st in status]
    out["status"] = status
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5, help="holding period in sessions")
    ap.add_argument("--ledger", default=LEDGER_FILE)
    args = ap.parse_args()

    picks = unique_picks(args.ledger)
    graded = grade(picks, args.horizon)
    graded.to_csv(GRADED_FILE, index=False)

    settled = graded[graded["was_right"].notna()]
    pending = (graded["status"] == "pending").sum()
    print(f"Graded {len(graded)} picks @ {args.horizon}-session horizon "
          f"({len(settled)} settled, {pending} still pending).")
    print(f"-> {GRADED_FILE}\n")

    show = graded[["date", "rank", "ticker", "agg_sent", "realized_return", "status"]].copy()
    show["date"] = show["date"].dt.date
    show["realized_return"] = show["realized_return"].map(
        lambda x: f"{x:+.2%}" if pd.notna(x) else "  pending")
    show["agg_sent"] = show["agg_sent"].map(lambda x: f"{x:+.3f}")
    print(show.to_string(index=False))

    if not settled.empty:
        hit = settled["was_right"].mean()
        avg = settled["realized_return"].mean()
        print(f"\nSettled-pick scorecard:  hit-rate {hit:.0%}  |  "
              f"avg return {avg:+.2%}  |  n={len(settled)}")
        print("Read each row's reason card in decision_ledger.csv to judge whether a "
              "right call was skill — or a wrong call was bad luck — not just the score.")
    else:
        print("\nNo picks have completed their holding horizon yet. Re-run later.")


if __name__ == "__main__":
    main()
