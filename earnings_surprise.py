"""
earnings_surprise.py — does the CHANGE in earnings-call tone predict drift?

The mechanism the level test lacked: markets react to *surprise*, not absolute
mood. A call that sounds positive when the prior call was already positive is no
news; a call that turns more positive than last quarter is. So we test the
quarter-over-quarter delta in tone, not the level.

Signal:  surprise(q) = finbert_sent(q) - finbert_sent(prior quarter, same ticker)
         (also tested: agg_earnings delta)
Target:  market-excess forward return after the transcript date (open->open,
         demeaned within the quarter cohort), horizons 1/5/20 sessions.

Power note: needs >=2 consecutive quarters per ticker, so the first quarter of
each ticker is dropped. With 8 fetched quarters we get up to 7 surprise cohorts
— far more than the 2 the level (drift) test had.

Run:  python earnings_surprise.py
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr, ttest_1samp

OUT_FILE = "results/earnings_surprise_results.csv"
HORIZONS = [1, 5, 20]
LEVELS = {"finbert_sent": "surprise_finbert", "agg_earnings": "surprise_agg"}


def q_sortkey(q: str) -> tuple[int, int]:
    """'Q3FY26' -> (26, 3) so quarters sort chronologically within a ticker."""
    m = re.match(r"Q(\d)FY(\d+)", q)
    return (int(m.group(2)), int(m.group(1))) if m else (0, 0)


def session_opens(tickers, start, end):
    px = yf.download(list(tickers), start=start, end=end, auto_adjust=True,
                     progress=False, group_by="ticker")
    out = {}
    for tk in tickers:
        try:
            s = px[tk]["Open"] if len(tickers) > 1 else px["Open"]
            out[tk] = s.dropna()
        except Exception:
            pass
    return out


def fwd_return(open_s, t_date, h):
    after = open_s[open_s.index.date > t_date]
    if len(after) < h + 1:
        return None
    entry, exit_ = after.iloc[0], after.iloc[h]
    if entry <= 0 or np.isnan(entry) or np.isnan(exit_):
        return None
    return exit_ / entry - 1.0


def main():
    ev = pd.read_csv("results/scored_earnings.csv")
    ev["t_date"] = pd.to_datetime(ev["transcript_date"]).dt.date
    ev["qkey"] = ev["quarter"].map(q_sortkey)

    # build QoQ surprise per ticker (delta vs the immediately prior quarter present)
    ev = ev.sort_values(["ticker", "qkey"]).reset_index(drop=True)
    for lvl, scol in LEVELS.items():
        ev[scol] = ev.groupby("ticker")[lvl].diff()

    n_q = ev["quarter"].nunique()
    surp = ev.dropna(subset=list(LEVELS.values())).copy()
    print(f"events: {len(ev)} across {ev['ticker'].nunique()} tickers, "
          f"{n_q} quarters {sorted(ev['quarter'].unique(), key=q_sortkey)}")
    print(f"surprise obs (>=2 consecutive q): {len(surp)} across "
          f"{surp['quarter'].nunique()} surprise cohorts")

    tickers = sorted(surp["ticker"].unique())
    start = min(surp["t_date"]).isoformat()
    end = (max(surp["t_date"]) + pd.Timedelta(days=45)).isoformat()
    print(f"fetching opens {start} .. {end} ...")
    opens = session_opens(tickers, start, end)

    for h in HORIZONS:
        surp[f"fwd_{h}"] = [
            fwd_return(opens[r.ticker], r.t_date, h) if r.ticker in opens else None
            for r in surp.itertuples()
        ]
    surp.to_csv(OUT_FILE, index=False)

    print("\n" + "=" * 80)
    print("EARNINGS-TONE SURPRISE (QoQ delta) vs POST-CALL DRIFT "
          "(market-excess, demeaned within quarter)")
    print("=" * 80)
    print(f"  {'signal':<18}{'horizon':<9}{'pooledIC':>9}{'p':>8}"
          f"{'meanIC(q)':>11}{'t':>7}{'p(q)':>8}{'LS top-bot':>12}{'n':>6}")

    for scol in LEVELS.values():
        for h in HORIZONS:
            col = f"fwd_{h}"
            sub = surp[["quarter", scol, col]].dropna().copy()
            if len(sub) < 8:
                continue
            sub["ret_resid"] = sub.groupby("quarter")[col].transform(lambda x: x - x.mean())
            # surprise is already a change; demean within cohort for cross-sectional rank
            sub["sig_resid"] = sub.groupby("quarter")[scol].transform(lambda x: x - x.mean())

            pooled = spearmanr(sub["sig_resid"], sub["ret_resid"])
            q_ics = []
            for _, g in sub.groupby("quarter"):
                if len(g) >= 5 and g["sig_resid"].nunique() > 1:
                    q_ics.append(spearmanr(g["sig_resid"], g["ret_resid"]).correlation)
            mean_ic = np.nanmean(q_ics) if q_ics else np.nan
            tval, pq = (ttest_1samp(q_ics, 0) if len(q_ics) >= 2 else (np.nan, np.nan))

            n3 = max(1, len(sub) // 3)
            ranked = sub.sort_values(scol)
            ls = ranked["ret_resid"].iloc[-n3:].mean() - ranked["ret_resid"].iloc[:n3].mean()

            print(f"  {scol:<18}{h:<9}{pooled.correlation:>+9.3f}{pooled.pvalue:>8.3f}"
                  f"{mean_ic:>+11.3f}{tval:>+7.2f}{pq:>8.3f}{ls:>+11.2%}{len(sub):>6}")

    print(f"\n  [caveat] {surp['quarter'].nunique()} surprise cohorts. p(q) is the "
          "honest read (t-test of per-quarter ICs);\n  pooledIC ignores cross-"
          "sectional clustering and overstates significance.")
    print(f"\n  Per-event detail -> {OUT_FILE}")  # results/earnings_surprise_results.csv


if __name__ == "__main__":
    main()
