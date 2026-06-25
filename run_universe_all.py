"""
run_universe_all.py — Nifty-50 cross-sectional out-of-sample sweep.

Pipeline (score-once / reweight-many, so we never re-run FinBERT per window):
  1. For each month JSON (articles_uni_<mon>.json), run FinBERT+LM ONCE and cache
     the raw scores to scored_uni_<mon>.csv (idempotent — skips if present).
  2. Build A/B sub-windows per month (A: news 1-12 -> move 12-15; B: 15-28 -> 28-31).
  3. Per window: reweight cached scores as-of the cutoff, aggregate per ticker to a
     `combined` signal, fetch forward returns + trailing momentum for all 50 names,
     and compute MARKET-RELATIVE cross-sectional metrics:
       - balanced directional accuracy on excess returns
       - rank IC (sentiment vs momentum), per-window mean + significance
       - decile long-short book (top-5 vs bottom-5 by signal)

With ~45 names/window the IC standard error is ~3x smaller than the 4-name test,
so a real (or absent) edge finally becomes testable.

Run (long; prefer background):  python run_universe_all.py
"""
import argparse
import dataclasses
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
import yfinance as yf

from sentiment_core import Config, load_models, score_articles, apply_weights, aggregate
from universe import NIFTY_50, TICKERS, MONTH_FETCH
try:
    from fetch_earnings import get_earnings_for_window
    _HAS_EARNINGS = True
except Exception:
    _HAS_EARNINGS = False

MOM_LOOKBACK = 5
DECILE = 5            # names per long / short leg
CACHE_COLS = ["id", "scope", "source", "date", "pos_prob", "neg_prob",
              "neu_prob", "sent", "news", "pos_count", "neg_count"]


# ---------- step 1: score each month once, cache ----------
def scored_month(mon, models):
    cache = f"results/scored_uni_{mon}.csv"
    if os.path.exists(cache):
        return pd.read_csv(cache)
    src = f"articles_uni_{mon}.json"
    if not os.path.exists(src):
        print(f"[{mon}] {src} missing -> skip"); return None
    df = pd.read_json(src)
    print(f"[{mon}] scoring {len(df)} articles (FinBERT+LM, one-time)...")
    scored = score_articles(df, models=models)
    scored[CACHE_COLS].to_csv(cache, index=False)
    print(f"[{mon}] cached -> {cache}")
    return scored[CACHE_COLS]


# ---------- windows ----------
def build_windows():
    wins = []
    months = {m[0]: m for m in MONTH_FETCH}
    for mon in months:
        y = 2026
        m = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6}[mon]
        a_cut = date(y, m, 12)
        wins.append((f"{mon}-A", mon, date(y, m, 1), a_cut, a_cut + timedelta(days=3)))
        if mon != "jun":                      # June B is in the future
            b_cut = date(y, m, 28)
            wins.append((f"{mon}-B", mon, date(y, m, 15), b_cut, b_cut + timedelta(days=3)))
    return wins


# ---------- prices: forward return + trailing momentum for the universe ----------
def universe_prices(cutoff, eval_end):
    start = pd.Timestamp(cutoff) - pd.Timedelta(days=MOM_LOOKBACK * 3 + 15)
    end = pd.Timestamp(eval_end) + pd.Timedelta(days=3)
    px = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                     progress=False, group_by="ticker")
    rows = []
    for tk in TICKERS:
        try:
            s = px[tk]["Close"].dropna()
        except (KeyError, TypeError):
            continue
        s.index = s.index.normalize()
        pre = s.index[s.index <= pd.Timestamp(cutoff)]
        post = s.index[s.index <= pd.Timestamp(eval_end)]
        if len(pre) < MOM_LOOKBACK + 1 or len(post) == 0 or pre[-1] == post[-1]:
            continue
        rows.append({"ticker": tk,
                     "actual_ret": float(s.loc[post[-1]] / s.loc[pre[-1]] - 1.0),
                     "momentum": float(s.loc[pre[-1]] / s.loc[pre[-1 - MOM_LOOKBACK]] - 1.0)})
    return pd.DataFrame(rows)


def long_short(g, sig, ret, k):
    g = g.dropna(subset=[sig, ret])
    if len(g) < 2 * k:
        return np.nan
    g = g.sort_values(sig)
    return float(g[ret].tail(k).mean() - g[ret].head(k).mean())


def balanced_accuracy(pred_pos, actual_pos):
    pred_pos, actual_pos = np.asarray(pred_pos), np.asarray(actual_pos)
    tp = np.sum(pred_pos & actual_pos); fn = np.sum(~pred_pos & actual_pos)
    tn = np.sum(~pred_pos & ~actual_pos); fp = np.sum(pred_pos & ~actual_pos)
    rp = tp / (tp + fn) if (tp + fn) else np.nan
    rn = tn / (tn + fp) if (tn + fp) else np.nan
    return float(np.nanmean([rp, rn])), rp, rn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recency-max-age", type=int, default=Config.recency_max_age,
                    help="article lookback window in days: articles older than this get "
                         "weight 0 at scoring time. Lets you replay a shorter news window "
                         "(e.g. 5) on cached data without re-fetching. Effective baseline is "
                         "the 14-day fetch cap (MAX_AGE_DAYS), even though the default is 60.")
    args = ap.parse_args()

    cfg = dataclasses.replace(Config(), recency_max_age=args.recency_max_age)
    print(f"[config] recency_max_age = {cfg.recency_max_age} days "
          f"(article lookback window)\n")
    models = load_models()

    # step 1: score/cache all months
    scored = {}
    for mon, _, _ in MONTH_FETCH:
        s = scored_month(mon, models)
        if s is not None:
            scored[mon] = s

    # steps 2-3: per-window cross-sectional metrics
    rows, per_window = [], []
    for label, mon, start, cutoff, eval_end in build_windows():
        if mon not in scored:
            continue
        df = scored[mon].copy()
        df["d"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["d"] >= start) & (df["d"] <= cutoff)]
        if df.empty:
            continue
        w = apply_weights(df, cfg, today=cutoff)
        agg = pd.DataFrame(aggregate(w, cfg, today=cutoff, tickers=TICKERS))
        # combined == agg_sent: the LM news leg has negative cross-sectional IC and
        # dilutes the signal (signal_search.py); a FinBERT news leg just duplicates
        # agg_sent (news_leg_experiment.py). So the predictive signal is FinBERT alone.
        agg["combined"] = agg["agg_sent"]

        if _HAS_EARNINGS:
            earnings_map = get_earnings_for_window(cutoff, cfg)
            agg["agg_earnings"] = agg["ticker"].map(earnings_map)  # NaN if no transcript
        else:
            agg["agg_earnings"] = np.nan

        px = universe_prices(cutoff, eval_end)
        g = agg.merge(px, on="ticker", how="inner").dropna(subset=["actual_ret"])
        # keep only names that actually had news (a non-zero signal contribution)
        g = g[(g["n_sent"] > 0) | (g["n_news"] > 0)]
        if len(g) < 2 * DECILE:
            print(f"[{label}] only {len(g)} usable names -> skip")
            continue

        g = g.copy(); g["window"] = label
        g["ret_resid"] = g["actual_ret"] - g["actual_ret"].mean()
        g["sig_resid"] = g["combined"] - g["combined"].mean()
        rows.append(g)

        ic_s = spearmanr(g["combined"], g["actual_ret"]).correlation
        ic_m = spearmanr(g["momentum"], g["actual_ret"]).correlation
        n_earn = g["agg_earnings"].notna().sum() if "agg_earnings" in g.columns else 0
        per_window.append({"window": label, "n": len(g),
                           "ic_sent": ic_s, "ic_mom": ic_m,
                           "ls_sent": long_short(g, "combined", "actual_ret", DECILE),
                           "ls_mom": long_short(g, "momentum", "actual_ret", DECILE)})
        print(f"[{label}] names={len(g):<3} IC_sent={ic_s:+.3f} IC_mom={ic_m:+.3f} "
              f"LS_sent={per_window[-1]['ls_sent']:+.2%}  earnings_cov={n_earn}/{len(g)}")

    df_all = pd.concat(rows, ignore_index=True)
    pw = pd.DataFrame(per_window)
    df_all.to_csv("results/universe_calls.csv", index=False)
    pw.to_csv("results/universe_windows.csv", index=False)
    # Dual-write: replace the database copies of both tables for this full run.
    try:
        import db_io
        if db_io.write_universe_calls(df_all):
            print(f"  (DB) wrote {len(df_all)} rows -> universe_calls table.")
        if db_io.write_universe_windows(pw):
            print(f"  (DB) wrote {len(pw)} rows -> universe_windows table.")
    except Exception as e:
        print(f"  (DB) universe write skipped: {e}")

    print("\n" + "=" * 72)
    print(f"NIFTY-50 CROSS-SECTIONAL SWEEP — {len(pw)} windows, "
          f"{len(df_all)} ticker-observations")
    print("=" * 72)

    g2 = df_all[df_all["sig_resid"] != 0]
    bal, rp, rn = balanced_accuracy(g2["sig_resid"] > 0, g2["ret_resid"] > 0)
    print(f"\n--- DIRECTION (excess; 50% = no skill) ---")
    print(f"  raw excess accuracy : {(( (g2['sig_resid']>0)==(g2['ret_resid']>0)).mean()):.1%}  (n={len(g2)})")
    print(f"  BALANCED accuracy   : {bal:.1%}  (outperf-recall {rp:.0%} / underperf-recall {rn:.0%})")

    def ic_report(col, resid, lbl):
        ics = pw[col].dropna()
        pooled = spearmanr(df_all[resid], df_all["ret_resid"]).correlation
        t, p = ttest_1samp(ics, 0.0) if len(ics) >= 2 else (np.nan, np.nan)
        ir = ics.mean() / ics.std(ddof=1) if ics.std(ddof=1) > 0 else np.nan
        print(f"  {lbl:<10} mean IC={ics.mean():+.3f} IR={ir:+.2f} t={t:+.2f} "
              f"p={p:.3f} | pooled IC={pooled:+.3f}")

    print("\n--- RANK IC ---")
    ic_report("ic_sent", "sig_resid", "sentiment")
    ic_report("ic_mom", "momentum" if "momentum" in df_all else "sig_resid", "momentum")

    def ls_report(col, lbl):
        s = pw[col].dropna()
        if len(s) < 2:
            print(f"  {lbl:<10} n/a"); return
        t, p = ttest_1samp(s, 0.0)
        print(f"  {lbl:<10} mean={s.mean():+.2%}/win win-rate={(s>0).mean():.0%} "
              f"t={t:+.2f} p={p:.3f} ({len(s)} windows)")

    print(f"\n--- DECILE LONG-SHORT (top-{DECILE} vs bottom-{DECILE}) ---")
    ls_report("ls_sent", "sentiment")
    ls_report("ls_mom", "momentum")

    sig_ic = pw["ic_sent"].dropna()
    t, p = ttest_1samp(sig_ic, 0.0) if len(sig_ic) >= 2 else (np.nan, np.nan)
    print("\n--- VERDICT ---")
    if pd.notna(p) and p < 0.05 and sig_ic.mean() > 0:
        print(f"  Significant positive cross-sectional IC (mean {sig_ic.mean():+.3f}, p={p:.3f}).")
    elif sig_ic.mean() > 0:
        print(f"  Positive but not significant (mean IC {sig_ic.mean():+.3f}, p={p:.3f}).")
    else:
        print(f"  No edge (mean IC {sig_ic.mean():+.3f}, p={p:.3f}).")
    print("\n  Detail -> results/universe_calls.csv, results/universe_windows.csv")


if __name__ == "__main__":
    main()
