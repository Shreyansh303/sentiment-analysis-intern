"""
earnings_drift.py — does earnings-call sentiment predict post-earnings drift?

Event study on scored_earnings.csv. For each (ticker, quarter) transcript:
  - ENTRY = first session OPEN on/after the day AFTER transcript_date
    (the transcript is public by transcript_date, so we enter next session —
    no look-ahead).
  - Forward return at horizons H in {1,5,20} sessions = open->open.
  - Market-excess return (ret_resid) = demean the forward return WITHIN the
    quarter cohort, matching run_universe_all's cross-sectional definition.

Signal tested: finbert_sent (FinBERT transcript mood) and agg_earnings (the
blended earnings score). IC = Spearman rank corr vs ret_resid, pooled across
all events and also as a mean of the two per-quarter ICs (the honest, very
low-power read — only 2 earnings seasons exist).

Run:  python earnings_drift.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr, ttest_1samp

EARN_FILE = "earnings_drift_results.csv"
HORIZONS = [1, 5, 20]
SIGNALS = ["finbert_sent", "agg_earnings"]


def session_opens(tickers, start, end):
    """{ticker -> Open series indexed by session date} over a padded window."""
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
    """Open->open return: enter at first open strictly AFTER t_date, exit h
    sessions later. None if not enough sessions exist."""
    after = open_s[open_s.index.date > t_date]
    if len(after) < h + 1:
        return None
    entry = after.iloc[0]
    exit_ = after.iloc[h]
    if entry <= 0 or np.isnan(entry) or np.isnan(exit_):
        return None
    return exit_ / entry - 1.0


def main():
    ev = pd.read_csv("scored_earnings.csv")
    ev["t_date"] = pd.to_datetime(ev["transcript_date"]).dt.date
    tickers = sorted(ev["ticker"].unique())
    start = (min(ev["t_date"])).isoformat()
    end = (max(ev["t_date"]) + pd.Timedelta(days=45)).isoformat()  # room for +20 sessions
    print(f"events: {len(ev)} across {len(tickers)} tickers, "
          f"quarters={sorted(ev['quarter'].unique())}")
    print(f"fetching opens {start} .. {end} ...")
    opens = session_opens(tickers, start, end)

    # attach forward returns at each horizon
    for h in HORIZONS:
        ev[f"fwd_{h}"] = [
            fwd_return(opens[r.ticker], r.t_date, h) if r.ticker in opens else None
            for r in ev.itertuples()
        ]

    ev.to_csv(EARN_FILE, index=False)

    print("\n" + "=" * 78)
    print("EARNINGS-CALL SENTIMENT vs POST-EARNINGS DRIFT (market-excess, "
          "demeaned within quarter)")
    print("=" * 78)
    print(f"  {'signal':<14}{'horizon':<9}{'pooledIC':>9}{'p':>8}"
          f"{'meanIC(q)':>11}{'LS top-bot':>12}{'n':>6}")

    for sig in SIGNALS:
        for h in HORIZONS:
            col = f"fwd_{h}"
            sub = ev[["quarter", sig, col]].dropna().copy()
            if len(sub) < 8:
                continue
            # market-excess return + demeaned signal, within each quarter cohort
            sub["ret_resid"] = sub.groupby("quarter")[col].transform(
                lambda x: x - x.mean())
            sub["sig_resid"] = sub.groupby("quarter")[sig].transform(
                lambda x: x - x.mean())

            pooled = spearmanr(sub["sig_resid"], sub["ret_resid"]).correlation
            # per-quarter IC, then mean + t (only ~2 quarters => tiny power)
            q_ics = []
            for _, g in sub.groupby("quarter"):
                if len(g) >= 5 and g["sig_resid"].nunique() > 1:
                    q_ics.append(spearmanr(g["sig_resid"], g["ret_resid"]).correlation)
            mean_ic = np.nanmean(q_ics) if q_ics else np.nan
            p = ttest_1samp(q_ics, 0).pvalue if len(q_ics) >= 2 else np.nan

            # long-short: top third minus bottom third by raw signal, on resid ret
            n3 = max(1, len(sub) // 3)
            ranked = sub.sort_values(sig)
            ls = ranked["ret_resid"].iloc[-n3:].mean() - ranked["ret_resid"].iloc[:n3].mean()

            print(f"  {sig:<14}{h:<9}{pooled:>+9.3f}"
                  f"{('%.3f' % spearmanr(sub['sig_resid'], sub['ret_resid']).pvalue):>8}"
                  f"{mean_ic:>+11.3f}{ls:>+11.2%}{len(sub):>6}")

    print("\n  [caveat] Only", ev["quarter"].nunique(), "earnings seasons exist "
          f"({sorted(ev['quarter'].unique())}). pooledIC pools events but they "
          "cluster by\n  quarter, so the effective sample is closer to 2 "
          "independent draws. Treat as exploratory.")
    print(f"\n  Per-event detail -> {EARN_FILE}")


if __name__ == "__main__":
    main()
