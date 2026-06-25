"""
run_combined_book.py — sector-neutral, signal-weighted long-short book combining
SENTIMENT SURPRISE with PRICE MOMENTUM. Re-runs on cached universe_calls.csv only
(no re-scoring / re-fetching).

Design (the Tier-1 + Tier-2 recommendations, in one book):
  - SENTIMENT SURPRISE: each name's `combined` sentiment minus its own trailing
    (prior-windows-only, expanding) mean. Captures "better/worse news than usual",
    which moves prices — and auto-removes the persistent negativity bias. Strictly
    leakage-free (uses only earlier windows).
  - SECTOR-NEUTRAL: within each window, subtract the sector mean from each alpha,
    so we bet stock-vs-peers, not sector-vs-sector.
  - Z-SCORE & COMBINE: standardise sentiment-surprise and momentum across the
    cross-section, then sum (equal weight) into one alpha.
  - SIGNAL-WEIGHTED, DOLLAR-NEUTRAL BOOK: weights proportional to alpha, demeaned
    so sum(w)=0, scaled to gross 1.0 (0.5 long / 0.5 short). Portfolio return per
    window = sum(w_i * actual_ret_i); the market cancels by construction.

Reports sentiment-only vs momentum-only vs COMBINED, gross and net of cost, with
significance — so we can see whether combining clears momentum-alone.
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, spearmanr

CALLS = "../results/universe_calls.csv"
COST_BPS = 10.0          # per-window round-trip cost on gross (assumes full turnover)

WINDOW_ORDER = ["jan-A", "jan-B", "feb-A", "feb-B", "mar-A", "mar-B",
                "apr-A", "apr-B", "may-A", "may-B", "jun-A"]

# Broad sector buckets (each >= 3 names so neutralisation is meaningful).
SECTOR = {
    "TCS.NS":"IT","INFY.NS":"IT","HCLTECH.NS":"IT","WIPRO.NS":"IT","TECHM.NS":"IT","LTIM.NS":"IT",
    "HDFCBANK.NS":"FIN","ICICIBANK.NS":"FIN","SBIN.NS":"FIN","KOTAKBANK.NS":"FIN","AXISBANK.NS":"FIN",
    "INDUSINDBK.NS":"FIN","BAJFINANCE.NS":"FIN","BAJAJFINSV.NS":"FIN","SHRIRAMFIN.NS":"FIN",
    "SBILIFE.NS":"FIN","HDFCLIFE.NS":"FIN",
    "RELIANCE.NS":"ENERGY","ONGC.NS":"ENERGY","BPCL.NS":"ENERGY","COALINDIA.NS":"ENERGY",
    "NTPC.NS":"ENERGY","POWERGRID.NS":"ENERGY",
    "MARUTI.NS":"AUTO","M&M.NS":"AUTO","TATAMOTORS.NS":"AUTO","EICHERMOT.NS":"AUTO",
    "HEROMOTOCO.NS":"AUTO","BAJAJ-AUTO.NS":"AUTO",
    "SUNPHARMA.NS":"PHARMA","DRREDDY.NS":"PHARMA","CIPLA.NS":"PHARMA","DIVISLAB.NS":"PHARMA",
    "APOLLOHOSP.NS":"PHARMA",
    "TATASTEEL.NS":"MATERIALS","JSWSTEEL.NS":"MATERIALS","HINDALCO.NS":"MATERIALS",
    "GRASIM.NS":"MATERIALS","ULTRACEMCO.NS":"MATERIALS","ASIANPAINT.NS":"MATERIALS",
    "HINDUNILVR.NS":"CONSUMER","ITC.NS":"CONSUMER","NESTLEIND.NS":"CONSUMER",
    "BRITANNIA.NS":"CONSUMER","TATACONSUM.NS":"CONSUMER","TITAN.NS":"CONSUMER",
    "LT.NS":"INDUSTRIALS","ADANIENT.NS":"INDUSTRIALS","ADANIPORTS.NS":"INDUSTRIALS",
    "BHARTIARTL.NS":"INDUSTRIALS",
}


def zscore(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def sector_neutralize(df, col):
    """Subtract the per-window sector mean, then z-score across the window."""
    neutral = df[col] - df.groupby(["window", "sector"])[col].transform("mean")
    return df.assign(_n=neutral).groupby("window")["_n"].transform(zscore)


def book_returns(df, alpha_col, gross=1.0):
    """Signal-weighted dollar-neutral portfolio return per window."""
    out = []
    for w, g in df.groupby("window"):
        a = g[alpha_col] - g[alpha_col].mean()       # enforce dollar-neutral
        denom = a.abs().sum()
        if denom == 0 or len(g) < 4:
            continue
        wt = a / denom * gross                        # sum|w| = gross
        ret = float((wt * g["actual_ret"]).sum())
        out.append({"window": w, "ret": ret, "turnover": float(wt.abs().sum())})
    return pd.DataFrame(out)


def stats(pnl, label, cost_bps):
    if pnl.empty or len(pnl) < 2:
        print(f"  {label:<22} n/a"); return
    gross = pnl["ret"]
    cost = pnl["turnover"] * cost_bps / 1e4
    net = gross - cost
    t, p = ttest_1samp(gross, 0.0)
    tn, pn = ttest_1samp(net, 0.0)
    ir = gross.mean() / gross.std(ddof=1) if gross.std(ddof=1) > 0 else np.nan
    print(f"  {label:<22} gross={gross.mean():+.2%}/win (t={t:+.2f}, p={p:.3f})  "
          f"net={net.mean():+.2%}/win (p={pn:.3f})  win={ (gross>0).mean():.0%}  IR={ir:+.2f}")


def main():
    # Read from the database first; fall back to the CSV if it's empty/unavailable.
    df = None
    try:
        import db_io
        df = db_io.read_universe_calls()
    except Exception as e:
        print(f"  (DB) universe_calls read skipped, using CSV: {e}")
    if df is None or df.empty:
        df = pd.read_csv(CALLS)
    df["sector"] = df["ticker"].map(SECTOR).fillna("OTHER")
    df["wo"] = df["window"].map({w: i for i, w in enumerate(WINDOW_ORDER)})
    df = df.sort_values(["ticker", "wo"]).reset_index(drop=True)

    # --- sentiment SURPRISE: combined minus own trailing (prior-only) mean ---
    df["sent_trail"] = (df.groupby("ticker")["combined"]
                          .transform(lambda s: s.expanding().mean().shift(1)))
    df["surprise"] = df["combined"] - df["sent_trail"]
    # windows with no history for a name -> no sentiment bet (neutral 0)
    df["surprise"] = df["surprise"].fillna(0.0)

    # --- sector-neutralize + z-score each raw alpha within window ---
    df["a_sent"] = sector_neutralize(df, "surprise")
    df["a_mom"] = sector_neutralize(df, "momentum")
    df["a_combined"] = df["a_sent"] + df["a_mom"]

    # --- earnings alpha (if agg_earnings present from fetch_earnings.py) ---
    has_earnings = (
        "agg_earnings" in df.columns and df["agg_earnings"].notna().any()
    )
    if has_earnings:
        # NaN = no transcript for that ticker-window; treat as neutral (0) before
        # sector-neutralization so those names shrink toward the cross-section mean
        # rather than being excluded from the book.
        df["agg_earnings_filled"] = df["agg_earnings"].fillna(0.0)
        df["a_earnings"] = sector_neutralize(df, "agg_earnings_filled")
        df["a_3factor"] = df["a_sent"] + df["a_mom"] + df["a_earnings"]
        earn_cov = df["agg_earnings"].notna().mean()

    # diversification check
    corr = df[["a_sent", "a_mom"]].corr().iloc[0, 1]

    print("=" * 78)
    print("SECTOR-NEUTRAL SIGNAL-WEIGHTED BOOK — sentiment-surprise + momentum")
    print(f"  {len(df)} obs | {df['window'].nunique()} windows | "
          f"alpha corr(sent_surprise, momentum) = {corr:+.2f}  (low = diversifying)")
    if has_earnings:
        print(f"  earnings transcript coverage = {earn_cov:.0%} of obs")
    print("=" * 78)

    print(f"\n--- PORTFOLIO P&L per book (cost {COST_BPS:.0f}bps/win on gross) ---")
    stats(book_returns(df, "a_sent"), "sentiment-surprise", COST_BPS)
    stats(book_returns(df, "a_mom"), "momentum", COST_BPS)
    stats(book_returns(df, "a_combined"), "COMBINED (2-factor)", COST_BPS)
    if has_earnings:
        stats(book_returns(df, "a_earnings"), "earnings-tone", COST_BPS)
        stats(book_returns(df, "a_3factor"), "3-FACTOR COMBINED", COST_BPS)

    # rank IC of each alpha vs forward return (pooled, market-relative)
    df["ret_resid"] = df.groupby("window")["actual_ret"].transform(lambda x: x - x.mean())
    print("\n--- POOLED RANK IC (alpha vs excess return) ---")
    ic_cols = [("a_sent", "sentiment-surprise"), ("a_mom", "momentum"),
               ("a_combined", "COMBINED (2-factor)")]
    if has_earnings:
        ic_cols += [("a_earnings", "earnings-tone"), ("a_3factor", "3-FACTOR COMBINED")]
    for col, lbl in ic_cols:
        ic = spearmanr(df[col], df["ret_resid"]).correlation
        print(f"  {lbl:<22} pooled IC = {ic:+.3f}")

    if has_earnings:
        corr3 = df[["a_sent", "a_mom", "a_earnings"]].corr()
        print(f"\n  earnings vs sent_surprise corr = {corr3.loc['a_earnings','a_sent']:+.2f}  "
              f"earnings vs momentum corr = {corr3.loc['a_earnings','a_mom']:+.2f}")

    # per-window combined return series for inspection
    bc = book_returns(df, "a_combined")
    bc.to_csv("../results/combined_book_pnl.csv", index=False)
    if has_earnings:
        b3 = book_returns(df, "a_3factor")
        b3.to_csv("../results/threefactor_book_pnl.csv", index=False)
    # Dual-write: replace the database copy (backtest_pnl table).
    try:
        import db_io
        if db_io.write_backtest_pnl(bc):
            print(f"  (DB) wrote {len(bc)} rows -> backtest_pnl table.")
    except Exception as e:
        print(f"  (DB) backtest_pnl write skipped: {e}")
    print("\n  Combined per-window P&L -> combined_book_pnl.csv")
    if has_earnings:
        print("  3-factor per-window P&L  -> threefactor_book_pnl.csv")


if __name__ == "__main__":
    main()
