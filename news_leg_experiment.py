"""
news_leg_experiment.py — would a FinBERT-scored "news" leg beat the LM one?

The shipped pipeline scores the news leg with the Loughran-McDonald dictionary
(`news` column) and the sentiment leg with FinBERT (`sent` column). signal_search
showed the LM news leg has NEGATIVE cross-sectional IC. This asks: if we score the
news leg with FinBERT instead — but KEEP its distinctive weighting (trusted sources
only, 10-day recency, macro-aware damping) — does it turn positive, and is it still
a DISTINCT signal from agg_sent or just a duplicate?

Fully offline: reuses cached per-article scores (scored_uni_<mon>.csv has both `sent`
and `news`) and the cached forward excess returns in universe_calls.csv. No FinBERT
re-run, no price download.

Three per-ticker variants, all from the SAME articles:
  sent     = w_sent | score=sent (FinBERT) | sent damping      -> current agg_sent
  news_lm  = w_news | score=news (LM)       | news damping      -> current agg_news
  news_fb  = w_news | score=sent (FinBERT)  | news damping      -> the experiment

Run:  python3 news_leg_experiment.py
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

warnings.simplefilter("ignore", category=RuntimeWarning)

from sentiment_core import Config, apply_weights, _weighted_inputs, _weighted_mean
from run_universe_all import build_windows
from universe import TICKERS   # full Nifty universe (~48 names), as run_universe_all uses

TARGET = "ret_resid"   # market-excess forward return, cached in universe_calls.csv
cfg = Config()

# cached forward returns / reference aggregates from the last universe sweep
calls = pd.read_csv("universe_calls.csv")


def agg_variant(wdf, tk, wcol, score_col, cut_col, sec, mac):
    v, w = _weighted_inputs(wdf, tk, wcol, score_col, cut_col, sec, mac)
    return _weighted_mean(v, w), len(v)


rows = []
for label, mon, start, cutoff, eval_end in build_windows():
    path = f"scored_uni_{mon}.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        continue
    df["d"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["d"] >= start) & (df["d"] <= cutoff)]
    if df.empty:
        continue
    w = apply_weights(df, cfg, today=cutoff)
    for tk in TICKERS:
        sent, n_s = agg_variant(w, tk, "w_sent", "sent", "cut_sent",
                                cfg.sent_sector_damp, cfg.sent_macro_damp)
        news_lm, n_n = agg_variant(w, tk, "w_news", "news", "cut_news",
                                   cfg.news_sector_damp, cfg.news_macro_damp)
        news_fb, _ = agg_variant(w, tk, "w_news", "sent", "cut_news",
                                 cfg.news_sector_damp, cfg.news_macro_damp)
        rows.append({"window": label, "ticker": tk, "n_sent": n_s, "n_news": n_n,
                     "sent": sent, "news_lm": news_lm, "news_fb": news_fb})

panel = pd.DataFrame(rows)
# blends
panel["combined_lm"] = 0.5 * panel["sent"] + 0.5 * panel["news_lm"]   # shipped
panel["combined_fb"] = 0.5 * panel["sent"] + 0.5 * panel["news_fb"]   # FinBERT-both

# join cached forward excess return
panel = panel.merge(calls[["window", "ticker", TARGET, "agg_sent", "agg_news"]],
                    on=["window", "ticker"], how="inner")
panel = panel[(panel["n_sent"] > 0) | (panel["n_news"] > 0)]

# ---- faithfulness check: recomputed sent/news_lm should match the cached sweep ----
chk = panel.dropna(subset=["agg_sent", "agg_news"])
d_sent = (chk["sent"] - chk["agg_sent"]).abs().max()
d_news = (chk["news_lm"] - chk["agg_news"]).abs().max()
print(f"faithfulness: max|sent-agg_sent|={d_sent:.4f}  max|news_lm-agg_news|={d_news:.4f}"
      f"  (small => re-aggregation matches the pipeline)\n")


def per_window_ic(col):
    ics = []
    for _, g in panel.groupby("window"):
        g = g.dropna(subset=[col, TARGET])
        if g[col].nunique() < 3 or len(g) < 5:
            continue
        ics.append(spearmanr(g[col], g[TARGET]).correlation)
    return np.array([x for x in ics if pd.notna(x)])


def report(col):
    ics = per_window_ic(col)
    t, p = ttest_1samp(ics, 0.0) if len(ics) >= 2 else (np.nan, np.nan)
    ir = ics.mean() / ics.std(ddof=1) if ics.std(ddof=1) > 0 else np.nan
    print(f"  {col:<14} meanIC={ics.mean():+.3f}  IR={ir:+.2f}  t={t:+.2f}  p={p:.3f}"
          f"  (n_win={len(ics)})")


print(f"panel: {len(panel)} obs, {panel['window'].nunique()} windows\n")
print("=== news leg: LM vs FinBERT scoring (news-leg weighting kept) ===")
for c in ["news_lm", "news_fb", "sent"]:
    report(c)
print("\n=== blends ===")
for c in ["combined_lm", "combined_fb"]:
    report(c)

# redundancy check: is news_fb a distinct signal or just agg_sent again?
corr = panel[["news_fb", "sent"]].dropna().corr().iloc[0, 1]
corr_lm = panel[["news_lm", "sent"]].dropna().corr().iloc[0, 1]
print(f"\ncorr(news_fb, sent) = {corr:+.3f}   corr(news_lm, sent) = {corr_lm:+.3f}")
print("  (high corr => FinBERT news leg is largely a duplicate of agg_sent)")

panel.to_csv("news_leg_experiment_panel.csv", index=False)
print("\nDetail -> news_leg_experiment_panel.csv")
