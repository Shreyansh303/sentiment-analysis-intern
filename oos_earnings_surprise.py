"""
oos_earnings_surprise.py — the out-of-sample gate for the earnings-surprise signal.

The signal (agg_earnings QoQ surprise -> 20d drift, go long high-surprise / short
low-surprise) was DISCOVERED on the full sample. This script freezes that choice
and asks the only question that matters: does it hold on cohorts it never saw?

Split (chronological, no shuffling):
  TRAIN  = Q2FY25 .. Q2FY26   (5 surprise cohorts — where the effect was found)
  TEST   = Q3FY26, Q4FY26     (2 newest cohorts — held out, cold)

Strategy is fixed from TRAIN: signal=surprise_agg, horizon=20 sessions, direction
= long top third / short bottom third (demeaned within cohort). We then read the
TEST cohorts' IC and long-short spread. No refitting.

Run:  python oos_earnings_surprise.py
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

SIGNAL = "surprise_agg"     # frozen choice
HORIZON = "fwd_20"          # frozen choice
TEST_COHORTS = {"Q3FY26", "Q4FY26"}


def qk(q):
    m = re.match(r"Q(\d)FY(\d+)", q)
    return (int(m.group(2)), int(m.group(1)))


def cohort_stats(df):
    """Per-cohort IC and long-short (top third minus bottom third) on resid return."""
    rows = []
    for q, g in df.groupby("quarter"):
        g = g.dropna(subset=[SIGNAL, HORIZON]).copy()
        if len(g) < 6:
            continue
        rr = g[HORIZON] - g[HORIZON].mean()
        sr = g[SIGNAL] - g[SIGNAL].mean()
        ic = spearmanr(sr, rr).correlation
        n3 = max(1, len(g) // 3)
        ranked = g.assign(rr=rr).sort_values(SIGNAL)
        ls = ranked["rr"].iloc[-n3:].mean() - ranked["rr"].iloc[:n3].mean()
        rows.append({"quarter": q, "n": len(g), "IC": ic, "LS_20d": ls})
    return pd.DataFrame(rows).sort_values("quarter", key=lambda s: s.map(qk))


def main():
    d = pd.read_csv("earnings_surprise_results.csv")
    d = d.dropna(subset=[SIGNAL, HORIZON])
    d["is_test"] = d["quarter"].isin(TEST_COHORTS)

    train, test = d[~d["is_test"]], d[d["is_test"]]
    st_tr, st_te = cohort_stats(train), cohort_stats(test)

    print("=" * 70)
    print("OOS GATE: agg_earnings surprise -> 20d drift (long high / short low)")
    print("=" * 70)
    print(f"\nTRAIN (discovery) cohorts: {sorted(train['quarter'].unique(), key=qk)}")
    print(st_tr.to_string(index=False, float_format=lambda x: f'{x:+.3f}'))
    print(f"  mean IC = {st_tr.IC.mean():+.3f} | mean LS = {st_tr.LS_20d.mean():+.2%}")

    print(f"\nTEST (held-out, cold) cohorts: {sorted(test['quarter'].unique(), key=qk)}")
    print(st_te.to_string(index=False, float_format=lambda x: f'{x:+.3f}'))
    print(f"  mean IC = {st_te.IC.mean():+.3f} | mean LS = {st_te.LS_20d.mean():+.2%}")

    # pooled OOS read: every test event's long/short bucket return
    n3 = max(1, len(test) // 3)
    tr = test.copy()
    tr["rr"] = tr.groupby("quarter")[HORIZON].transform(lambda x: x - x.mean())
    ranked = tr.sort_values(SIGNAL)
    longs, shorts = ranked["rr"].iloc[-n3:], ranked["rr"].iloc[:n3]
    pooled_ic = spearmanr(
        tr[SIGNAL] - tr.groupby("quarter")[SIGNAL].transform("mean"), tr["rr"]).correlation
    print(f"\nPOOLED OOS (n={len(test)} held-out events):")
    print(f"  IC = {pooled_ic:+.3f}")
    print(f"  long basket {longs.mean():+.2%}  vs  short basket {shorts.mean():+.2%}  "
          f"-> spread {longs.mean()-shorts.mean():+.2%}")

    print("\n" + "-" * 70)
    # Honest bar: the TRADEABLE edge (long-short) must survive, not just the IC sign.
    sign_holds = (st_te.IC > 0).all()
    edge_holds = st_te.LS_20d.mean() > 0 and (longs.mean() - shorts.mean()) > 0
    ic_decay = 1 - st_te.IC.mean() / st_tr.IC.mean() if st_tr.IC.mean() else float("nan")
    print(f"IC decay train->test: {ic_decay:.0%}  | "
          f"test mean long-short: {st_te.LS_20d.mean():+.2%}")
    if sign_holds and edge_holds and ic_decay < 0.5:
        print("VERDICT: signal HOLDS out-of-sample — worth a real forward test.")
    elif sign_holds and not edge_holds:
        print("VERDICT: INCONCLUSIVE / mostly faded. IC sign survives but the")
        print("         tradeable long-short edge does not, and IC decays sharply.")
        print("         Classic in-sample inflation; needs LIVE forward quarters,")
        print("         not more historical slicing.")
    else:
        print("VERDICT: signal does NOT hold OOS — in-sample edge was likely noise.")


if __name__ == "__main__":
    main()
