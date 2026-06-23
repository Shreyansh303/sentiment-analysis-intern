"""
Sensitivity sweep — "kill the magic numbers, or justify them."

For each hard-coded threshold in the pipeline (divergence 0.20, the +/-0.25 label
cutoffs, the FinBERT impact buckets 0.85/0.60/0.40, the scope damping 0.5/0.2/...,
the 0.05 cut), we vary ONE parameter across a range, rebuild the entire signal
panel from cached raw scores, join forward returns, and report whether IC / P&L /
labels are STABLE across that range.

The honest reading:
  - "Results hold from 0.15 to 0.25" (IC barely moves)  -> the number isn't magic.
  - IC swings wildly with the threshold                  -> the result is an artifact.

Two classes of parameter, handled differently:
  * SIGNAL params (damping, impact buckets, cut) change the numeric agg_sent/agg_news
    -> we report how IC and long-short P&L move.
  * LABEL params (divergence, label_cutoff) only re-bin the SAME numbers into
    categories -> numeric IC is invariant (we show that), so we report how many
    cells flip label vs the default config.

Requires scored_history.csv (written by pipeline_nifty.py). With one day of history
the IC columns read n/a — honestly so — but the signal-range and label-flip columns
are informative immediately, and IC/P&L fill in as the panel grows.

Usage:
    python sensitivity.py                      # sweep all params, signal=agg_sent
    python sensitivity.py --signal agg_news
    python sensitivity.py --param divergence_thr   # just one
"""
import argparse
import dataclasses
from datetime import date

import numpy as np
import pandas as pd

import backtest as bt
from sentiment_core import Config, apply_weights, aggregate, load_scored_history

# Each entry: field name -> (sweep values, kind). kind in {"signal","label"}.
SWEEPS = {
    "divergence_thr":   ([0.10, 0.15, 0.20, 0.25, 0.30], "label"),
    "label_cutoff":     ([0.15, 0.20, 0.25, 0.30, 0.35], "label"),
    "impact_high_thr":  ([0.75, 0.80, 0.85, 0.90, 0.95], "signal"),
    "impact_med_thr":   ([0.50, 0.55, 0.60, 0.65, 0.70], "signal"),
    "impact_low_thr":   ([0.30, 0.35, 0.40, 0.45, 0.50], "signal"),
    "sent_sector_damp": ([0.3, 0.4, 0.5, 0.6, 0.7], "signal"),
    "sent_macro_damp":  ([0.1, 0.2, 0.3, 0.4, 0.5], "signal"),
    "news_sector_damp": ([0.7, 0.8, 0.9, 1.0], "signal"),
    "news_macro_damp":  ([0.5, 0.6, 0.7, 0.8, 0.9], "signal"),
    "cut_threshold":    ([0.02, 0.05, 0.08, 0.10, 0.15], "signal"),
}


def panel_for_cfg(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Replay the whole panel for a given Config over every snapshot day in the
    cached scores. Bootstrap disabled here — we only need point aggregates."""
    cfg = dataclasses.replace(cfg, bootstrap_n=0)
    rows = []
    for snap, g in scored.groupby("snap_date"):
        d = date.fromisoformat(str(snap))
        weighted = apply_weights(g.copy(), cfg, d)
        rows.extend(aggregate(weighted, cfg, d))
    panel = pd.DataFrame(rows)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    return panel


def evaluate_panel(panel, opens, signal, default_labels, cost_bps):
    """Return a metrics dict for one swept config."""
    df = bt.attach_forward_returns(panel, opens)
    out = {}
    for name in bt.HORIZONS:
        ic, n, _, p = bt.info_coefficient(df, signal, name)
        out[f"ic_{name}"] = ic
        out[f"n_{name}"] = n
        out[f"p_{name}"] = p
        pnl = bt.long_short_pnl(df, signal, name, cost_bps)
        out[f"pnl_{name}"] = pnl["cum"].iloc[-1] if not pnl.empty else np.nan
    out["sig_min"] = panel[signal].min()
    out["sig_max"] = panel[signal].max()
    # label stability vs the default config
    merged = panel.merge(default_labels, on=["date", "ticker"], suffixes=("", "_def"))
    out["labels_flipped"] = int((merged["label"] != merged["label_def"]).sum())
    out["n_labels"] = len(merged)
    return out


def fmt(x, pct=False, plus=False):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "  n/a"
    if pct:
        return f"{x:+.2%}" if plus else f"{x:.2%}"
    return f"{x:+.3f}" if plus else f"{x:.3f}"


def run_sweep(field, values, kind, scored, opens, signal, default_labels, cost_bps, default_val):
    print(f"\n=== {field}  (default {default_val}, kind={kind}) ===")
    header = (f"  {'value':>8} {'IC_1d':>8} {'p_1d':>6} {'IC_5d':>8} "
              f"{'P&L_5d':>9} {'sig[min,max]':>16} {'Δlabels':>9}")
    print(header)
    base = Config()
    for v in values:
        cfg = dataclasses.replace(base, **{field: v})
        panel = panel_for_cfg(scored, cfg)
        m = evaluate_panel(panel, opens, signal, default_labels, cost_bps)
        star = "*" if (pd.notna(m["p_fwd_1d"]) and m["p_fwd_1d"] < 0.05) else " "
        mark = " <-default" if v == default_val else ""
        rng = f"[{m['sig_min']:+.2f},{m['sig_max']:+.2f}]"
        print(f"  {v:>8} {fmt(m['ic_fwd_1d'],plus=True):>8}{star} "
              f"{fmt(m['p_fwd_1d']):>6} {fmt(m['ic_fwd_5d'],plus=True):>8} "
              f"{fmt(m['pnl_fwd_5d'],pct=True,plus=True):>9} {rng:>16} "
              f"{m['labels_flipped']:>4}/{m['n_labels']:<4}{mark}")
    if kind == "label":
        print("  (label param: numeric IC is invariant by construction — the signal it "
              "moves is the\n   categorical view, so read the Δlabels column, not IC.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="agg_sent",
                    help="numeric column to score: agg_sent | agg_news | divergence")
    ap.add_argument("--param", default=None, help="sweep just one field (default: all)")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    args = ap.parse_args()

    scored = load_scored_history()
    n_days = scored["snap_date"].nunique()
    print(f"Scored history: {len(scored)} article-rows across {n_days} snapshot day(s). "
          f"Signal under test = {args.signal}.")
    if n_days < 2:
        print("[!] Only one snapshot day cached. The sweep runs and the signal-range / "
              "Δlabels\n    columns are meaningful now, but IC/P&L need ~15-20 days of "
              "history to read.\n    Keep running pipeline_nifty.py daily.")

    # Prices + the default-config labels (the reference the sweep is measured against).
    base_panel = panel_for_cfg(scored, Config())
    opens = bt.fetch_prices(base_panel["ticker"].unique(),
                            base_panel["date"].min(), base_panel["date"].max())
    default_labels = base_panel[["date", "ticker", "label"]].rename(columns={"label": "label_def"})

    items = SWEEPS.items() if args.param is None else [(args.param, SWEEPS[args.param])]
    base = Config()
    for field, (values, kind) in items:
        run_sweep(field, values, kind, scored, opens, args.signal,
                  default_labels, args.cost_bps, getattr(base, field))


if __name__ == "__main__":
    main()
