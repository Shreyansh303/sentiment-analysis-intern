"""
signal_search.py — disciplined hunt for a tradable signal in the cross-sectional panel.

Reads universe_calls.csv (529 ticker-observations over 11 windows, already joined to
forward excess returns) and, for each CANDIDATE signal, reports the honest
unit-of-analysis statistics:
  - per-window rank IC, its mean, info ratio (mean/std), and a t-test on the 11
    window ICs (this is the correct test; pooling 529 rows overstates significance)
  - pooled IC for reference
  - a top-k / bottom-k long-short book mean per window + its t-test

It also tests CONDITIONING (high-news-count names only) and SIGN FLIPS (is the
signal contrarian?), and prints a multiple-testing caveat because we try many things.

Run:  python signal_search.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

PANEL = "../results/universe_calls.csv"
K = 5  # names per long/short leg

df = pd.read_csv(PANEL)
# ret_resid = market-relative (excess) forward return; that's our target.
TARGET = "ret_resid"
windows = list(df["window"].unique())


def per_window_ic(d, sig, target=TARGET):
    ics = []
    for _, g in d.groupby("window"):
        g = g.dropna(subset=[sig, target])
        if g[sig].nunique() < 3 or len(g) < 5:
            continue
        ics.append(spearmanr(g[sig], g[target]).correlation)
    return np.array([x for x in ics if pd.notna(x)])


def long_short_per_window(d, sig, target=TARGET, k=K):
    vals = []
    for _, g in d.groupby("window"):
        g = g.dropna(subset=[sig, target])
        if len(g) < 2 * k:
            continue
        g = g.sort_values(sig)
        vals.append(g[target].tail(k).mean() - g[target].head(k).mean())
    return np.array(vals)


def report(name, d, sig, k=K):
    ics = per_window_ic(d, sig)
    ls = long_short_per_window(d, sig, k=k)
    pooled = spearmanr(*[d.dropna(subset=[sig, TARGET])[c] for c in (sig, TARGET)]).correlation
    if len(ics) >= 2:
        t_ic, p_ic = ttest_1samp(ics, 0.0)
        ir = ics.mean() / ics.std(ddof=1) if ics.std(ddof=1) > 0 else np.nan
    else:
        t_ic = p_ic = ir = np.nan
    if len(ls) >= 2:
        t_ls, p_ls = ttest_1samp(ls, 0.0)
    else:
        t_ls = p_ls = np.nan
    print(f"{name:<34} meanIC={ics.mean():+.3f} IR={ir:+.2f} t={t_ic:+.2f} p={p_ic:.3f} "
          f"| pooledIC={pooled:+.3f} | LS={ls.mean():+.3%}/win t={t_ls:+.2f} p={p_ls:.3f} "
          f"| n_win={len(ics)}")
    return {"name": name, "mean_ic": ics.mean(), "ir": ir, "p_ic": p_ic,
            "ls_mean": ls.mean(), "p_ls": p_ls, "n_win": len(ics)}


print(f"panel: {len(df)} obs, {len(windows)} windows, target={TARGET} (market-excess fwd return)\n")

results = []
print("=== RAW CANDIDATE SIGNALS ===")
for sig in ["combined", "agg_sent", "agg_news", "momentum"]:
    if sig in df.columns:
        results.append(report(sig, df, sig))

print("\n=== SIGN FLIP (is it contrarian / mean-reverting?) ===")
for sig in ["combined", "agg_sent"]:
    if sig in df.columns:
        d = df.copy(); d["_neg"] = -d[sig]
        results.append(report(f"-{sig} (contrarian)", d, "_neg"))

print("\n=== CONDITIONING: only names with enough news (low_n == False) ===")
if "low_n" in df.columns:
    hi = df[~df["low_n"].astype(bool)]
    print(f"  ({len(hi)}/{len(df)} obs survive the high-news filter)")
    for sig in ["combined", "agg_sent", "agg_news"]:
        if sig in hi.columns:
            results.append(report(f"{sig} | high-news", hi, sig))

print("\n=== CONDITIONING: top-tercile by news count per window ===")
if "n_news" in df.columns:
    thr = df.groupby("window")["n_news"].transform(lambda s: s.quantile(2 / 3))
    busy = df[df["n_news"] >= thr]
    print(f"  ({len(busy)}/{len(df)} obs survive)")
    for sig in ["combined", "agg_news"]:
        if sig in busy.columns:
            results.append(report(f"{sig} | busy-tercile", busy, sig, k=3))

print("\n=== COMBINING WITH MOMENTUM (z-score blend) ===")
if {"combined", "momentum"}.issubset(df.columns):
    d = df.copy()
    for c in ["combined", "momentum"]:
        d[c + "_z"] = d.groupby("window")[c].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) else 0.0)
    for w in [0.5, 0.3]:
        d["blend"] = w * d["combined_z"] + (1 - w) * d["momentum_z"]
        results.append(report(f"blend {w:.1f}*sent+{1-w:.1f}*mom", d, "blend"))

res = pd.DataFrame(results)
res.to_csv("../results/signal_search_results.csv", index=False)

print("\n" + "=" * 72)
n_tests = len(res)
hits = res[(res["p_ic"] < 0.05) | (res["p_ls"] < 0.05)]
print(f"Ran {n_tests} candidate signals. Bonferroni-ish threshold for 1 false "
      f"positive: p < {0.05 / n_tests:.4f}")
print(f"Nominal p<0.05 hits: {len(hits)} (expected by chance at this many tests: "
      f"{0.05 * n_tests:.1f})")
if len(hits):
    print(hits[["name", "mean_ic", "p_ic", "ls_mean", "p_ls"]].to_string(index=False))
print("\nDetail -> signal_search_results.csv")
