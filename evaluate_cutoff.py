"""
Point-in-time, leakage-free evaluation of the Nifty sentiment/news model.

The premise (a clean out-of-sample test):
  1.  The model is shown ONLY articles dated on or before a hard cutoff
      (default 2026-06-05). A strict filter + assertions guarantee that no
      byte of information published after the cutoff ever reaches the model.
  2.  As of that cutoff the model emits a per-ticker stock-movement prediction
      (UP / DOWN / NEUTRAL) from the FinBERT + Loughran-McDonald analysts.
  3.  We then fetch the ACTUAL price path over the evaluation window
      (cutoff -> end, default 2026-06-05 -> 2026-06-18) and score the
      predictions: directional hit-rate, per-ticker detail, and a rank check.

Why this differs from backtest.py: that harness validates an *accumulating
daily panel* of forward returns. This script is a single, self-contained
as-of-date experiment with an explicit information barrier — the thing you
run to answer "if the model only knew what it knew on June 5, was it right?".

Usage:
    python evaluate_cutoff.py
    python evaluate_cutoff.py --cutoff 2026-06-05 --eval-end 2026-06-18
    python evaluate_cutoff.py --deadband 0.05 --basis close
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

import numpy as np
import pandas as pd

from sentiment_core import (
    Config, load_models, score_articles, apply_weights, aggregate,
)
from universe import TICKERS   # full Nifty 50, consistent with run_universe_all

ARTICLES_FILE = "articles_fetched.json"


# ----------------------------------------------------------------------------
# Step 1 — STRICT information barrier
# ----------------------------------------------------------------------------
def load_articles_with_cutoff(path: str, cutoff: date, start: date | None = None) -> pd.DataFrame:
    """Load articles and DROP everything published after `cutoff` (and, if
    `start` is given, anything before it — to bound the lookback window).

    Two independent guarantees that no future information leaks in:
      (a) we filter on the article's own publication date, and
      (b) we re-assert the post-filter max date <= cutoff and abort otherwise.
    The article date is normalised to YYYY-MM-DD before comparison so a stray
    timestamp/timezone suffix can't sneak a June-6 story through as "June 5".
    """
    df = pd.read_json(path)
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    art_date = pd.to_datetime(df["date"], errors="coerce").dt.date

    if art_date.isna().any():
        bad = df.loc[art_date.isna(), "id"].tolist()
        raise SystemExit(f"[FATAL] unparseable article dates, refusing to run: {bad}")

    n_total = len(df)
    keep = art_date <= cutoff
    if start is not None:
        keep &= art_date >= start
    visible = df[keep].copy().reset_index(drop=True)
    excluded = df[~keep]

    # No articles survived the filter -> nothing to predict on. Fail loudly with
    # the available date span so the cutoff/start can be corrected (otherwise the
    # barrier check below blows up on a NaN max_visible with an opaque TypeError).
    if visible.empty:
        lo, hi = art_date.min(), art_date.max()
        window = f"cutoff <= {cutoff}" + (f" and start >= {start}" if start else "")
        raise SystemExit(
            f"[FATAL] no articles match {window}: all {n_total} articles on disk "
            f"fall outside it (available span {lo} .. {hi}). "
            f"Pick a cutoff on/after {lo}."
        )

    # Hard barrier: re-derive the max visible date and refuse to proceed if
    # anything past the cutoff survived. Belt-and-suspenders on purpose.
    max_visible = pd.to_datetime(visible["date"]).dt.date.max()
    if max_visible > cutoff:
        raise SystemExit(
            f"[FATAL] cutoff barrier breached: a visible article is dated "
            f"{max_visible} > cutoff {cutoff}. Aborting to avoid look-ahead."
        )

    print(f"=== STEP 1: INFORMATION BARRIER (cutoff = {cutoff}) ===")
    if start is not None:
        print(f"  Lookback window        : {start} .. {cutoff}")
    print(f"  Total articles on disk : {n_total}")
    print(f"  Visible (<= {cutoff}) : {len(visible)}")
    print(f"  Excluded (>  {cutoff}) : {len(excluded)}  <- withheld from the model")
    if len(excluded):
        ex_by_day = excluded["date"].value_counts().sort_index()
        print("    withheld by day: " +
              ", ".join(f"{d}:{c}" for d, c in ex_by_day.items()))
    print(f"  Visible date span      : {visible['date'].min()} -> {visible['date'].max()}")
    return visible


# ----------------------------------------------------------------------------
# Step 2 — the model's prediction, as of the cutoff
# ----------------------------------------------------------------------------
def direction_from_signal(combined: float, deadband: float) -> str:
    if combined > deadband:
        return "UP"
    if combined < -deadband:
        return "DOWN"
    return "NEUTRAL"


def generate_predictions(visible: pd.DataFrame, cfg: Config, cutoff: date,
                         deadband: float, models=None) -> pd.DataFrame:
    """Score the visible articles ONCE (FinBERT + LM), weight them as-of the
    cutoff (so recency is anchored to June 5, not today), aggregate per ticker,
    and turn each aggregate into a directional call.

    The directional signal is agg_sent (FinBERT mood) alone: the LM news leg has
    negative cross-sectional IC and only dilutes it (signal_search.py /
    news_leg_experiment.py). A small dead-band abstains (NEUTRAL) near zero.
    """
    print("\n=== STEP 2: MODEL PREDICTION (as of cutoff) ===")
    scored = score_articles(visible, models=models if models is not None else load_models())
    # `today=cutoff` is the load-bearing line: recency decay is measured from
    # June 5, and recency_weight() clamps any age<0, so the model is frozen at
    # the cutoff's information state.
    weighted = apply_weights(scored, cfg, today=cutoff)
    rows = aggregate(weighted, cfg, today=cutoff, tickers=TICKERS)

    out = []
    for r in rows:
        # combined == agg_sent: the LM news leg has negative IC and dilutes the
        # signal (see signal_search.py / news_leg_experiment.py). FinBERT alone.
        combined = r["agg_sent"]
        out.append({
            "ticker": r["ticker"],
            "agg_sent": r["agg_sent"],
            "agg_news": r["agg_news"],
            "combined": round(combined, 4),
            "prediction": direction_from_signal(combined, deadband),
            "n_sent": r["n_sent"],
            "n_news": r["n_news"],
            "low_n": r["low_n"],
        })
    pred = pd.DataFrame(out)

    for _, r in pred.iterrows():
        warn = " (!) low-n" if r["low_n"] else ""
        print(f"  {r['ticker']:<13} combined={r['combined']:+.3f} "
              f"(sent={r['agg_sent']:+.2f}, news={r['agg_news']:+.2f}) "
              f"-> PREDICT {r['prediction']:<7}{warn}")
    print(f"  (dead-band = +/-{deadband}: |combined| below this abstains as NEUTRAL)")
    return pred


# ----------------------------------------------------------------------------
# Step 3 — actual realised movement over the evaluation window
# ----------------------------------------------------------------------------
def fetch_actual_moves(tickers, cutoff: date, eval_end: date, basis: str) -> pd.DataFrame:
    """Realised return of each ticker over (cutoff, eval_end], using daily
    auto-adjusted prices from yfinance.

      entry = `basis` price on the last session ON OR BEFORE the cutoff
              (the price you could transact at knowing only cutoff-day info)
      exit  = `basis` price on the last session ON OR BEFORE eval_end

    basis = 'close' (default) or 'open'. Returns one row per ticker with the
    actual session dates used, so the realised window is fully auditable.
    """
    import yfinance as yf

    print(f"\n=== STEP 3: ACTUAL MOVEMENT ({cutoff} -> {eval_end}, basis={basis}) ===")
    start = pd.Timestamp(cutoff) - pd.Timedelta(days=7)
    end = pd.Timestamp(eval_end) + pd.Timedelta(days=3)
    px = yf.download(list(tickers), start=start, end=end,
                     auto_adjust=True, progress=False, group_by="ticker")

    col = "Close" if basis == "close" else "Open"
    rows = []
    for tk in tickers:
        try:
            s = (px[tk][col] if len(tickers) > 1 else px[col]).dropna()
        except (KeyError, TypeError):
            print(f"  {tk:<13} no price data")
            rows.append({"ticker": tk, "actual_ret": np.nan})
            continue
        s.index = s.index.normalize()
        on_or_before_cut = s.index[s.index <= pd.Timestamp(cutoff)]
        on_or_before_end = s.index[s.index <= pd.Timestamp(eval_end)]
        if len(on_or_before_cut) == 0 or len(on_or_before_end) == 0:
            print(f"  {tk:<13} insufficient sessions in window")
            rows.append({"ticker": tk, "actual_ret": np.nan})
            continue
        d0, d1 = on_or_before_cut[-1], on_or_before_end[-1]
        p0, p1 = float(s.loc[d0]), float(s.loc[d1])
        ret = p1 / p0 - 1.0
        actual_dir = "UP" if ret > 0 else "DOWN" if ret < 0 else "FLAT"
        print(f"  {tk:<13} {d0.date()} {p0:>9.2f} -> {d1.date()} {p1:>9.2f}  "
              f"= {ret:+.2%}  ({actual_dir})")
        rows.append({
            "ticker": tk, "entry_date": d0.date().isoformat(),
            "exit_date": d1.date().isoformat(), "entry_px": round(p0, 2),
            "exit_px": round(p1, 2), "actual_ret": ret, "actual_dir": actual_dir,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Step 4 — grade the predictions
# ----------------------------------------------------------------------------
def report_accuracy(pred: pd.DataFrame, actual: pd.DataFrame):
    from scipy.stats import spearmanr

    df = pred.merge(actual, on="ticker", how="inner")
    df = df.dropna(subset=["actual_ret"])

    print("\n=== STEP 4: PREDICTION ACCURACY ===")
    header = (f"  {'ticker':<13}{'predict':<9}{'actual':<8}{'ret':>9}"
              f"{'combined':>10}  result")
    print(header)
    print("  " + "-" * (len(header) - 2))

    graded = df[df["prediction"] != "NEUTRAL"].copy()
    correct = 0
    for _, r in df.iterrows():
        if r["prediction"] == "NEUTRAL":
            res = "— (abstained)"
        else:
            hit = r["prediction"] == r["actual_dir"]
            correct += int(hit)
            res = "HIT  ✓" if hit else "MISS ✗"
        print(f"  {r['ticker']:<13}{r['prediction']:<9}{r['actual_dir']:<8}"
              f"{r['actual_ret']:>+8.2%}{r['combined']:>+10.3f}  {res}")

    n_called = len(graded)
    print()
    if n_called:
        acc = correct / n_called
        print(f"  Directional accuracy : {correct}/{n_called} = {acc:.0%}  "
              f"(NEUTRAL abstentions excluded)")
    else:
        print("  Directional accuracy : n/a (model abstained on every name)")

    # Rank check: does a higher combined signal line up with a higher actual
    # return? With only 4 names this is illustrative, not significant.
    sub = df[["combined", "actual_ret"]].dropna()
    if len(sub) >= 3 and sub["combined"].nunique() > 1:
        ic, p = spearmanr(sub["combined"], sub["actual_ret"])
        print(f"  Signal/return rank IC : {ic:+.2f} (p={p:.2f}, n={len(sub)})")

    print(f"\n  [caveat] n={len(df)} tickers — this is a single-window, small-sample "
          "read.\n  Treat it as a sanity check on the cutoff design, not proof of edge.")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default="2026-06-05",
                    help="information barrier; model sees only articles <= this date")
    ap.add_argument("--start", default=None,
                    help="optional lower bound; restrict the news lookback to >= this date")
    ap.add_argument("--eval-end", default="2026-06-18",
                    help="end of the realised-movement window")
    ap.add_argument("--deadband", type=float, default=0.05,
                    help="|combined| below this abstains (NEUTRAL)")
    ap.add_argument("--basis", choices=["close", "open"], default="close",
                    help="price basis for the realised return")
    ap.add_argument("--articles", default=ARTICLES_FILE)
    ap.add_argument("--out", default="cutoff_eval_results.csv")
    args = ap.parse_args()

    cutoff = date.fromisoformat(args.cutoff)
    eval_end = date.fromisoformat(args.eval_end)
    if eval_end <= cutoff:
        sys.exit("[FATAL] --eval-end must be after --cutoff")

    start = date.fromisoformat(args.start) if args.start else None
    cfg = Config()
    visible = load_articles_with_cutoff(args.articles, cutoff, start)
    pred = generate_predictions(visible, cfg, cutoff, args.deadband)
    actual = fetch_actual_moves(TICKERS, cutoff, eval_end, args.basis)
    result = report_accuracy(pred, actual)

    result.to_csv(args.out, index=False)
    # Dual-write: replace this (cutoff, eval_end) run in the database.
    try:
        import db_io
        if db_io.write_cutoff_eval(result, cutoff, eval_end):
            print(f"  (DB) wrote {len(result)} rows -> cutoff_eval table.")
    except Exception as e:
        print(f"  (DB) cutoff_eval write skipped: {e}")
    print(f"\n  Full results written to {args.out}")


if __name__ == "__main__":
    main()
