"""
Forward-return validation harness for the sentiment/news panel.

Reads snapshots.csv (the accumulating daily panel written by pipeline_nifty.py),
joins each (date, ticker) signal to the stock's FORWARD return, and reports:
  - Information Coefficient (Spearman rank corr between signal and forward return)
  - Hit-rate (fraction of signed signals that called direction correctly)
  - A toy long-short P&L curve (long top bucket / short bottom), net of cost.

Leakage rule: a snapshot dated D uses news available through D. We assume we can
only act at the NEXT session's open, so forward returns are measured open-to-open
starting from the next session. The signal date itself never touches the return
window -> no same-day leakage.

Usage:
    python backtest.py                       # all signals, default 10 bps cost
    python backtest.py --signal divergence   # which column to test
    python backtest.py --cost-bps 15         # round-trip cost in basis points
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

warnings.simplefilter("ignore", category=FutureWarning)

SNAPSHOT_FILE = "snapshots.csv"
HORIZONS = {"fwd_1d": 1, "fwd_5d": 5}   # trading-day horizons, measured open-to-open


def load_panel(path=SNAPSHOT_FILE):
    # Read from the database first; fall back to the CSV if empty/unavailable.
    try:
        import db_io
        panel = db_io.read_snapshots()
        if panel is not None and not panel.empty:
            return panel
    except Exception as e:
        print(f"  (DB) snapshots read skipped, using CSV: {e}")
    panel = pd.read_csv(path, parse_dates=["date"])
    panel["date"] = panel["date"].dt.normalize()
    return panel


def fetch_prices(tickers, start, end):
    """Daily OHLC for all tickers. Pad the window so the longest forward
    horizon is covered, plus slack for weekends/holidays."""
    start = pd.Timestamp(start) - pd.Timedelta(days=5)
    end = pd.Timestamp(end) + pd.Timedelta(days=max(HORIZONS.values()) * 2 + 10)
    px = yf.download(
        list(tickers), start=start, end=end,
        auto_adjust=True, progress=False, group_by="ticker",
    )
    # Normalize to a tidy {ticker -> open series indexed by session date}.
    opens = {}
    for tk in tickers:
        try:
            s = px[tk]["Open"] if len(tickers) > 1 else px["Open"]
        except (KeyError, TypeError):
            continue
        opens[tk] = s.dropna()
    return opens


def forward_return(open_series, signal_date, horizon):
    """Open-to-open forward return that begins at the FIRST session strictly
    after signal_date. Returns np.nan if not enough future sessions exist yet."""
    sessions = open_series.index
    future = sessions[sessions > signal_date]
    if len(future) < horizon + 1:
        return np.nan
    entry = open_series.loc[future[0]]        # next session's open (entry)
    exit_ = open_series.loc[future[horizon]]  # `horizon` sessions later
    return float(exit_ / entry - 1.0)


def attach_forward_returns(panel, opens):
    rows = []
    for _, r in panel.iterrows():
        tk = r["ticker"]
        if tk not in opens:
            continue
        out = r.to_dict()
        for name, h in HORIZONS.items():
            out[name] = forward_return(opens[tk], r["date"], h)
        rows.append(out)
    return pd.DataFrame(rows)


def info_coefficient(df, signal, ret_col):
    """Spearman IC plus a significance read. Returns (ic, n, t_stat, p_value).
    t_stat = ic * sqrt((n-2)/(1-ic^2)) tells you if the IC is distinguishable
    from zero; |t| > ~2 (p < 0.05) is the rough "this is real, not luck" line."""
    sub = df[[signal, ret_col]].dropna()
    if len(sub) < 3 or sub[signal].nunique() < 2:
        return np.nan, len(sub), np.nan, np.nan
    ic, p_value = spearmanr(sub[signal], sub[ret_col])
    n = len(sub)
    t_stat = ic * np.sqrt((n - 2) / (1 - ic ** 2)) if abs(ic) < 1 else np.nan
    return ic, n, t_stat, p_value


def hit_rate(df, signal, ret_col):
    """Fraction of non-zero signals whose sign matches the forward return."""
    sub = df[[signal, ret_col]].dropna()
    sub = sub[sub[signal] != 0]
    if sub.empty:
        return np.nan, 0
    correct = np.sign(sub[signal]) == np.sign(sub[ret_col])
    return float(correct.mean()), len(sub)


def long_short_pnl(df, signal, ret_col, cost_bps):
    """Per-date dollar-neutral long-short: long the top signal bucket, short the
    bottom. With few names the 'bucket' is the single best/worst. Each leg pays
    half the round-trip cost. Returns a per-date P&L series."""
    cost = cost_bps / 1e4
    daily = []
    for dt, g in df.groupby("date"):
        g = g.dropna(subset=[signal, ret_col])
        if len(g) < 2:
            continue
        g = g.sort_values(signal)
        n_side = max(1, len(g) // 3)          # top/bottom third (>=1 name)
        longs = g.tail(n_side)
        shorts = g.head(n_side)
        leg = longs[ret_col].mean() - shorts[ret_col].mean()
        # /2: average of the long and short legs; cost is the round-trip drag.
        daily.append({"date": dt, "ret": leg / 2 - cost})
    if not daily:
        return pd.DataFrame(columns=["date", "ret", "cum"])
    pnl = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)
    pnl["cum"] = (1 + pnl["ret"]).cumprod() - 1
    return pnl


def long_only_pnl(df, ret_col, cost_bps):
    """Equal-weight long-only basket return per date (the 'always-long' benchmark).
    One-way cost only — you're not shorting anything."""
    cost = cost_bps / 2e4
    daily = []
    for dt, g in df.groupby("date"):
        g = g.dropna(subset=[ret_col])
        if g.empty:
            continue
        daily.append({"date": dt, "ret": float(g[ret_col].mean()) - cost})
    if not daily:
        return pd.DataFrame(columns=["date", "ret", "cum"])
    pnl = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)
    pnl["cum"] = (1 + pnl["ret"]).cumprod() - 1
    return pnl


def max_drawdown(pnl):
    """Worst peak-to-trough decline of the equity curve (a negative number).
    Answers 'how much pain would I have sat through' — the part cum P&L hides."""
    if pnl.empty:
        return np.nan
    equity = 1 + pnl["cum"]            # equity curve starting at ~1.0
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def momentum_signal(panel, opens, lookback=5):
    """Pure price-momentum: trailing `lookback`-session return as of the signal
    date. Uses prices THROUGH the signal date only (the forward window starts the
    next session), so it's leakage-free on the same footing as the sentiment signal."""
    vals = []
    for _, r in panel.iterrows():
        s = opens.get(r["ticker"])
        if s is None:
            vals.append(np.nan)
            continue
        past = s.index[s.index <= r["date"]]
        if len(past) < lookback + 1:
            vals.append(np.nan)
            continue
        vals.append(float(s.loc[past[-1]] / s.loc[past[-1 - lookback]] - 1.0))
    return vals


def add_baseline_signals(df, opens, seed=0, mom_lookback=5):
    """Attach baseline signal columns to the returns frame:
      bl_long     — always-long (constant +1; a long-only benchmark)
      bl_random   — random labels (the null: a signal with no information)
      bl_momentum — pure price momentum (a real, non-sentiment competitor)
    Returns (df, list-of-(label, column, kind))."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["bl_long"] = 1.0
    df["bl_random"] = rng.uniform(-1.0, 1.0, size=len(df))
    df["bl_momentum"] = momentum_signal(df, opens, lookback=mom_lookback)
    specs = [
        ("always-long", "bl_long", "long_only"),
        ("random-label", "bl_random", "long_short"),
        (f"momentum({mom_lookback}d)", "bl_momentum", "long_short"),
    ]
    return df, specs


def report_signal(df, label, signal, kind, cost_bps):
    """One row per horizon: IC (+t,p), hit-rate, and P&L for a single signal.
    `kind` picks long-only vs long-short P&L construction."""
    for name in HORIZONS:
        ic, n_ic, t_stat, p_val = info_coefficient(df, signal, name)
        hr, n_hr = hit_rate(df, signal, name)
        pnl = (long_only_pnl(df, name, cost_bps) if kind == "long_only"
               else long_short_pnl(df, signal, name, cost_bps))
        ic_s = f"{ic:+.3f}" if pd.notna(ic) else "  n/a"
        t_s = f"{t_stat:+.2f}" if pd.notna(t_stat) else " n/a"
        hr_s = f"{hr:5.1%}" if pd.notna(hr) else "  n/a"
        pnl_s = f"{pnl['cum'].iloc[-1]:+.2%}" if not pnl.empty else "  n/a"
        sig = "*" if pd.notna(p_val) and p_val < 0.05 else " "
        print(f"  {label:<14} {name:<7}  IC={ic_s}{sig} (n={n_ic:<3}) t={t_s}  "
              f"hit={hr_s} (n={n_hr:<3})  cumP&L={pnl_s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="agg_sent",
                    help="panel column to test: agg_sent | agg_news | divergence")
    ap.add_argument("--cost-bps", type=float, default=10.0,
                    help="round-trip transaction cost in basis points")
    ap.add_argument("--snapshots", default=SNAPSHOT_FILE)
    ap.add_argument("--no-baselines", action="store_true",
                    help="skip the always-long / random / momentum comparison")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the random-label baseline")
    args = ap.parse_args()

    panel = load_panel(args.snapshots)
    n_days = panel["date"].nunique()
    print(f"Panel: {len(panel)} rows | {n_days} day(s) | "
          f"{panel['ticker'].nunique()} tickers | signal = {args.signal}")
    if n_days < 2:
        print("\n[!] Only one snapshot date so far. The harness runs, but IC and "
              "P&L need MULTIPLE days of history to mean anything.\n"
              "    Keep running pipeline_nifty.py daily — come back once you have "
              "~15-20 sessions.")

    opens = fetch_prices(panel["ticker"].unique(), panel["date"].min(), panel["date"].max())
    df = attach_forward_returns(panel, opens)

    print("\n=== INFORMATION COEFFICIENT (Spearman) & HIT-RATE ===")
    for name in HORIZONS:
        ic, n_ic, t_stat, p_val = info_coefficient(df, args.signal, name)
        hr, n_hr = hit_rate(df, args.signal, name)
        ic_s = f"{ic:+.3f}" if pd.notna(ic) else "  n/a"
        hr_s = f"{hr:5.1%}" if pd.notna(hr) else "  n/a"
        t_s = f"{t_stat:+.2f}" if pd.notna(t_stat) else " n/a"
        p_s = f"{p_val:.3f}" if pd.notna(p_val) else " n/a"
        sig = " *" if pd.notna(p_val) and p_val < 0.05 else "  "  # * = p<0.05
        print(f"  {name:<7}  IC={ic_s} (n={n_ic:<4})  t={t_s}  p={p_s}{sig}  "
              f"hit-rate={hr_s} (n={n_hr})")
    print("  (* = IC significant at p<0.05, i.e. unlikely to be luck)")

    print(f"\n=== TOY LONG-SHORT P&L (cost {args.cost_bps:.0f} bps round-trip) ===")
    for name in HORIZONS:
        pnl = long_short_pnl(df, args.signal, name, args.cost_bps)
        if pnl.empty:
            print(f"  {name:<7}  no tradable dates yet")
            continue
        total = pnl["cum"].iloc[-1]
        win = (pnl["ret"] > 0).mean()
        sharpe = (pnl["ret"].mean() / pnl["ret"].std() * np.sqrt(252)) \
            if pnl["ret"].std() > 0 else np.nan
        sh_s = f"{sharpe:.2f}" if pd.notna(sharpe) else "n/a"
        mdd = max_drawdown(pnl)
        mdd_s = f"{mdd:.2%}" if pd.notna(mdd) else "n/a"
        print(f"  {name:<7}  cum P&L={total:+.2%}  max drawdown={mdd_s}  "
              f"day win-rate={win:.1%}  ann.Sharpe={sh_s}  ({len(pnl)} periods)")
        out = f"pnl_{args.signal}_{name}.csv"
        pnl.to_csv(out, index=False)
        # Dual-write: replace this (signal, horizon) curve in the database.
        try:
            import db_io
            db_io.write_backtest_pnl_run(pnl, args.signal, name)
        except Exception as e:
            print(f"           (DB) pnl-run write skipped: {e}")
        print(f"           -> cumulative curve written to {out}")

    if not args.no_baselines:
        print("\n=== BASELINE COMPARISON (does the sentiment signal beat anything?) ===")
        print("  An IC of 0.05 means nothing in a vacuum. The sentiment signal must "
              "clear:\n  always-long (free beta), random labels (the null), and price "
              "momentum (a real rival).")
        df_bl, specs = add_baseline_signals(df, opens, seed=args.seed)
        report_signal(df_bl, f"SENTIMENT[{args.signal}]", args.signal, "long_short", args.cost_bps)
        for label, col, kind in specs:
            report_signal(df_bl, label, col, kind, args.cost_bps)
        print("  (* = IC significant at p<0.05. Watch for the sentiment signal failing "
              "to beat random/momentum — that's the honest read.)")


if __name__ == "__main__":
    main()
