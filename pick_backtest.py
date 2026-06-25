"""
pick_backtest.py — the honest gate for the long pick list.

A polished pick list looks convincing even when it's worthless. This asks the only
question that matters: would buying the top-N by agg_sent each day have BEATEN simply
holding the whole Nifty 50 basket — and does any edge survive out-of-sample?

It builds the SAME thing pick_top.py ships (long-only top-N, equal weight), measures
its forward return per date, and compares it to the always-long index benchmark. Then
it splits the history chronologically (train = older half, test = newer half) and
reports whether the edge holds on the held-out half. No refitting between the two.

Leakage rule is inherited from backtest.py: signal on date D, enter next session's
open, hold `horizon` sessions.

Usage:
    python pick_backtest.py                 # top 5, horizons 1d & 5d
    python pick_backtest.py --top 3
    python pick_backtest.py --split-date 2026-06-23   # force the train/test boundary
"""
import argparse

import numpy as np
import pandas as pd

from backtest import (
    fetch_prices, attach_forward_returns, HORIZONS,
    long_only_pnl, info_coefficient,
)

SIGNAL = "agg_sent"


def load_panel_csv(path):
    """Read snapshots.csv directly (not load_panel's DB-first path) so the backtest
    evaluates the SAME cross-sections the live picks come from; the DB copy has drifted."""
    panel = pd.read_csv(path, parse_dates=["date"])
    panel["date"] = panel["date"].dt.normalize()
    if "low_n" in panel:
        panel["low_n"] = panel["low_n"].map(
            lambda v: str(v).strip().lower() in ("true", "1"))
    return panel


def topn_pnl(df, ret_col, n, cost_bps, include_low_n):
    """Per-date equal-weight return of the top-N names by agg_sent (long-only).
    One-way cost only. Mirrors what pick_top.py would have you buy."""
    cost = cost_bps / 2e4
    daily = []
    for dt, g in df.groupby("date"):
        g = g.dropna(subset=[SIGNAL, ret_col])
        if not include_low_n:
            g = g[~g["low_n"]]
        if len(g) < n:
            continue
        top = g.sort_values(SIGNAL).tail(n)
        daily.append({"date": dt, "ret": float(top[ret_col].mean()) - cost})
    if not daily:
        return pd.DataFrame(columns=["date", "ret", "cum"])
    pnl = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)
    pnl["cum"] = (1 + pnl["ret"]).cumprod() - 1
    return pnl


def summarize(df, ret_col, n, cost_bps, include_low_n, label):
    """Print top-N vs index for one slice of dates and one horizon."""
    picks = topn_pnl(df, ret_col, n, cost_bps, include_low_n)
    index = long_only_pnl(df, ret_col, cost_bps)
    ic, n_ic, _, p = info_coefficient(df, SIGNAL, ret_col)
    if picks.empty or index.empty:
        print(f"  {label:<14} {ret_col:<7}  not enough names/dates to trade")
        return None
    edge = picks["ret"].mean() - index["ret"].mean()
    ic_s = f"{ic:+.3f}" if pd.notna(ic) else " n/a"
    sig = "*" if pd.notna(p) and p < 0.05 else " "
    print(f"  {label:<14} {ret_col:<7}  top{n}/day={picks['ret'].mean():+.3%}  "
          f"index={index['ret'].mean():+.3%}  edge={edge:+.3%}  "
          f"IC={ic_s}{sig} (n={n_ic})  [{len(picks)} days]")
    return edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--include-low-n", action="store_true")
    ap.add_argument("--split-date", default=None,
                    help="train/test boundary (YYYY-MM-DD); default = median date")
    ap.add_argument("--snapshots", default="snapshots.csv")
    args = ap.parse_args()

    panel = load_panel_csv(args.snapshots)
    dates = sorted(panel["date"].unique())
    print(f"Panel: {len(panel)} rows | {len(dates)} day(s) | "
          f"{panel['ticker'].nunique()} tickers | pick = top {args.top} by {SIGNAL}")

    opens = fetch_prices(panel["ticker"].unique(), panel["date"].min(), panel["date"].max())
    df = attach_forward_returns(panel, opens)

    print(f"\n=== FULL SAMPLE: top-{args.top} long-only vs always-long index "
          f"(cost {args.cost_bps:.0f} bps) ===")
    for name in HORIZONS:
        summarize(df, name, args.top, args.cost_bps, args.include_low_n, "FULL")

    # Chronological OOS split — older half trains, newer half is held out cold.
    if len(dates) < 4:
        print("\n[!] Fewer than 4 snapshot days — an OOS split is meaningless. Keep "
              "running pipeline_nifty.py daily; this gate needs ~15-20 sessions before "
              "the train/test read means anything.")
        return

    split = pd.Timestamp(args.split_date) if args.split_date else dates[len(dates) // 2]
    train, test = df[df["date"] < split], df[df["date"] >= split]
    print(f"\n=== OOS SPLIT at {pd.Timestamp(split).date()} "
          f"(train {train['date'].nunique()}d / test {test['date'].nunique()}d) ===")
    test_edges = {}
    for slice_df, lbl in ((train, "TRAIN"), (test, "TEST(cold)")):
        for name in HORIZONS:
            e = summarize(slice_df, name, args.top, args.cost_bps, args.include_low_n, lbl)
            if lbl.startswith("TEST"):
                test_edges[name] = e

    print("\n" + "-" * 70)
    holds = [e for e in test_edges.values() if e is not None]
    if holds and all(e > 0 for e in holds):
        print("VERDICT: picks beat the index on the held-out half at every horizon — "
              "promising. Confirm with more LIVE sessions before trusting it.")
    elif holds:
        print("VERDICT: the edge does NOT hold out-of-sample — at one or more horizons "
              "the top-N picks failed to beat simply holding the index.")
        print("         Consistent with the project's prior finding that this signal is "
              "near coin-flip. The pick list is fine as a transparency demo, but is not "
              "a validated way to make money.")
    else:
        print("VERDICT: not enough held-out data to judge. Keep accumulating sessions.")


if __name__ == "__main__":
    main()
