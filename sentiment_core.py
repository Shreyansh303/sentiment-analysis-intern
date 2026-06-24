"""
Shared core for the Nifty sentiment/news pipeline.

Everything that used to live as module-level constants + inline loops in
pipeline_nifty.py now lives here as:
  - a Config dataclass that holds EVERY magic number as a named, swept-able default
  - score_articles(): the expensive FinBERT + Loughran-McDonald pass. Run ONCE.
    It writes raw class probabilities (pos/neg/neu) so that downstream choices
    like impact buckets are cheap derivations, not re-inferences.
  - apply_weights() / aggregate(): the cheap, parameterised pass that turns raw
    probs into per-ticker aggregates + labels for a GIVEN Config.

Splitting "score once / re-weight many times" is what makes the sensitivity
sweep (sensitivity.py) tractable: we never re-run the transformer per parameter.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from datetime import date

import numpy as np
import pandas as pd

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
SECTOR_OF = {"TCS.NS": "SECTOR_IT", "INFY.NS": "SECTOR_IT"}

SOURCE_TRUST = {
    "screener.in / company release": 1.0, "upstox.com": 0.7, "Upstox": 0.7,
    "businesstoday.in (Equirus note)": 0.7, "msn.com (Antique Stock Broking)": 0.7,
    "cnbc.com": 0.8, "univest.in": 0.5, "businesstoday.in": 0.7,
    "tickertape.in / media reports": 0.6, "goodreturns.in": 0.6, "Goodreturns": 0.6,
    "bbntimes.com": 0.5, "pbs.org (AP)": 0.9, "indiainfoline.com": 0.6, "India Infoline": 0.6,
    "business-standard.com (Reuters)": 0.9, "business-standard.com": 0.8, "Business Standard": 0.8,
    "business-standard.com (PTI)": 0.8,
    "Reuters": 0.9, "Bloomberg": 0.9, "Bloomberg.com": 0.9,
    "Financial Times": 0.9, "WSJ": 0.9, "The Economic Times": 0.8,
    "ET Telecom": 0.7, "CNBC": 0.8, "CNBC TV18": 0.8,
    "Moneycontrol.com": 0.8, "Moneycontrol": 0.8, "Livemint": 0.8,
    "Mint": 0.8, "The Hindu": 0.8, "BusinessLine": 0.8,
    "The Times of India": 0.7, "The Indian Express": 0.7, "The New Indian Express": 0.7,
    "NDTV Profit": 0.8, "DD News": 0.8, "News On AIR": 0.8,
    "Yahoo Finance": 0.7, "Zacks": 0.5, "Simply Wall St.": 0.5,
    "simplywall.st": 0.5, "Value Research": 0.7, "TechCrunch": 0.7,
    "Investopedia": 0.7, "MSN": 0.6, "msn.com": 0.6,
}


@dataclass
class Config:
    """Every tunable number in one place. Defaults reproduce the original
    pipeline exactly; sensitivity.py perturbs one field at a time."""
    # recency half-lives (days)
    sent_half_life: int = 3
    news_half_life: int = 10
    recency_max_age: int = 60          # hard cutoff: older articles get weight 0

    # scope damping
    sent_sector_damp: float = 0.5
    sent_macro_damp: float = 0.2
    news_sector_damp: float = 0.9
    news_macro_damp: float = 0.7

    # FinBERT impact buckets (directional confidence = max(pos,neg))
    impact_high_thr: float = 0.85
    impact_med_thr: float = 0.60
    impact_low_thr: float = 0.40
    impact_w_high: float = 1.0
    impact_w_med: float = 0.6
    impact_w_low: float = 0.3

    # gating / cuts
    news_trust_gate: float = 0.6       # news analyst zeroes sources below this trust
    cut_threshold: float = 0.05        # drop article if final weight below this
    default_trust: float = 0.5

    # statistical honesty
    bootstrap_n: int = 1000            # resamples for the CI band (0 disables)
    low_n_threshold: int = 5           # cells with fewer contributing articles are flagged

    # earnings transcript scoring
    earnings_half_life: int = 45       # transcripts stay relevant much longer than news
    earnings_max_age: int = 90         # hard stale cutoff in days
    earnings_chunk_decay: float = 0.85 # position weight decay per FinBERT chunk
    earnings_finbert_w: float = 0.6    # composite weight on FinBERT sentiment
    earnings_lm_w: float = 0.4         # composite weight on LM polarity
    earnings_hedge_penalty: float = 0.3 # penalty per unit of (uncertainty+modal) ratio

    def as_dict(self):
        return asdict(self)


def source_trust(s, cfg: Config):
    """Look up trust, stripping fetch_articles.py's ' (via gnews-rss)' /
    ' (via yfinance)' harvester suffix so live sources match the table."""
    s = str(s)
    clean = re.sub(r"\s*\(via [^)]+\)\s*$", "", s).strip()
    if s in SOURCE_TRUST:
        return SOURCE_TRUST[s]
    if clean in SOURCE_TRUST:
        return SOURCE_TRUST[clean]
    s_lower, clean_lower = s.lower(), clean.lower()
    for k, v in SOURCE_TRUST.items():
        if k.lower() in (s_lower, clean_lower):
            return v
    return cfg.default_trust


def recency_weight(d, half_life, today: date, max_age: int):
    age = max(0, (today - date.fromisoformat(d)).days)   # clamp: no future-dated boost
    return 0.0 if age > max_age else 0.5 ** (age / half_life)


def preprocess_text_for_lm(text):
    """Clean text strictly for the bag-of-words Loughran-McDonald lexicon:
    lowercase, strip numbers, strip punctuation."""
    text_clean = str(text).lower()
    text_clean = re.sub(r"\d+", " ", text_clean)
    text_clean = re.sub(r"[^\w\s]", " ", text_clean)
    return text_clean


# ----------------------------------------------------------------------------
# Expensive pass: run the models ONCE, persist raw probabilities.
# ----------------------------------------------------------------------------
def load_models():
    """Lazy-import the heavy ML stack so cheap consumers (sweep on cached scores)
    don't pay the import cost."""
    import torch  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import pysentiment2 as ps

    print("Loading FinBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    lm = ps.LM()
    return tokenizer, model, lm


def score_articles(df: pd.DataFrame, models=None) -> pd.DataFrame:
    """Add raw, parameter-FREE scores to df: pos_prob/neg_prob/neu_prob (FinBERT),
    sent (= pos-neg), news (LM polarity), pos_count/neg_count. Impact buckets are
    deliberately NOT computed here — they depend on Config and are derived later."""
    import torch
    import torch.nn.functional as F

    if models is None:
        models = load_models()
    tokenizer, model, lm = models

    pos, neg, neu, news, posc, negc = [], [], [], [], [], []
    for _, row in df.iterrows():
        text = row["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        # ProsusAI/finbert labels: 0=Positive, 1=Negative, 2=Neutral
        p, n, u = float(probs[0]), float(probs[1]), float(probs[2])
        pos.append(p); neg.append(n); neu.append(u)

        score = lm.get_score(lm.tokenize(preprocess_text_for_lm(text)))
        news.append(float(score.get("Polarity", 0.0)))
        posc.append(int(score.get("Positive", 0)))
        negc.append(int(score.get("Negative", 0)))

    out = df.copy()
    out["pos_prob"] = np.round(pos, 4)
    out["neg_prob"] = np.round(neg, 4)
    out["neu_prob"] = np.round(neu, 4)
    out["sent"] = np.round(np.array(pos) - np.array(neg), 4)
    out["news"] = np.round(news, 4)
    out["pos_count"] = posc
    out["neg_count"] = negc
    return out


SCORED_HISTORY_FILE = "scored_history.csv"
_SCORED_COLS = ["id", "scope", "source", "date", "pos_prob", "neg_prob",
                "neu_prob", "sent", "news", "pos_count", "neg_count"]


def save_scored_history(df: pd.DataFrame, snap_date: date, path=SCORED_HISTORY_FILE):
    """Persist the raw (parameter-free) scores for this snapshot day, keyed by
    snap_date, idempotent per day. This is what lets sensitivity.py replay the
    parameter sweep across REAL history instead of just today — recency depends on
    the snapshot date, so we store the article date and re-derive weights later.

    Dual-write: writes both the database (article_scores) and the CSV file."""
    rec = df[_SCORED_COLS].copy()
    rec.insert(0, "snap_date", snap_date.isoformat())
    if os.path.exists(path):
        prev = pd.read_csv(path)
        prev = prev[prev["snap_date"] != snap_date.isoformat()]
        rec = pd.concat([prev, rec], ignore_index=True)
    rec.to_csv(path, index=False)
    try:
        import db_io
        db_io.upsert_article_scores(df, snap_date)
    except Exception as e:
        print(f"[db] scored-history DB write skipped: {e}")
    return rec


def load_scored_history(path=SCORED_HISTORY_FILE) -> pd.DataFrame:
    """Read from the database first (the new source of truth); fall back to the
    CSV file if the DB is empty or unavailable."""
    try:
        import db_io
        df = db_io.read_scored_history()
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f"[db] scored-history DB read skipped, using CSV: {e}")
    return pd.read_csv(path)


def score_transcript(text: str, models=None, chunk_size: int = 450) -> dict:
    """Score a full earnings call transcript for management tone.

    Parameter-free like score_articles — all tunable weights live in Config and
    are applied downstream. Results are cached in scored_earnings.csv.

    Returns:
      finbert_sent    position-weighted mean of (pos_prob - neg_prob) across chunks.
                      Earlier chunks (management remarks) outweigh later Q&A.
      lm_polarity     LM Polarity score on full text.
      lm_uncertainty  LM Uncertainty word fraction (hedging signal).
      lm_modal        LM (StrongModal + WeakModal) word fraction.
      n_chunks        number of FinBERT chunks scored.
      agg_earnings    composite: 0.6*finbert_sent + 0.4*lm_polarity
                      - 0.3*(lm_uncertainty + lm_modal), clipped to [-1, 1].
                      Uses default Config weights; callers can override downstream.
    """
    import torch
    import torch.nn.functional as F

    if models is None:
        models = load_models()
    tokenizer, model, lm = models

    # FinBERT: chunk transcript (max ~450 words ≈ safe under 512 subword limit),
    # weight chunks by position so prepared remarks dominate over Q&A.
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    chunk_sents = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        chunk_sents.append(float(probs[0]) - float(probs[1]))  # pos - neg

    if chunk_sents:
        pos_weights = [0.85 ** i for i in range(len(chunk_sents))]
        total_w = sum(pos_weights)
        finbert_sent = sum(s * w for s, w in zip(chunk_sents, pos_weights)) / total_w
    else:
        finbert_sent = 0.0

    # LM pass on full text: polarity + uncertainty + modal hedging
    clean = preprocess_text_for_lm(text)
    tokens = lm.tokenize(clean)
    score = lm.get_score(tokens)
    lm_polarity = float(score.get("Polarity", 0.0))
    n_tokens = max(len(tokens), 1)
    lm_uncertainty = int(score.get("Uncertainty", 0)) / n_tokens
    lm_modal = (int(score.get("StrongModal", 0)) + int(score.get("WeakModal", 0))) / n_tokens

    cfg = Config()  # use default weights for the parameter-free cached score
    hedging = lm_uncertainty + lm_modal
    raw = cfg.earnings_finbert_w * finbert_sent + cfg.earnings_lm_w * lm_polarity - cfg.earnings_hedge_penalty * hedging
    agg_earnings = float(np.clip(raw, -1.0, 1.0))

    return {
        "finbert_sent": round(finbert_sent, 4),
        "lm_polarity": round(lm_polarity, 4),
        "lm_uncertainty": round(lm_uncertainty, 4),
        "lm_modal": round(lm_modal, 4),
        "n_chunks": len(chunk_sents),
        "agg_earnings": round(agg_earnings, 4),
    }


# ----------------------------------------------------------------------------
# Cheap pass: everything below depends on Config and is swept.
# ----------------------------------------------------------------------------
def impact_label(pos_prob, neg_prob, cfg: Config):
    """Directional confidence -> bucket. A strongly-neutral article is low impact,
    so neu_prob is excluded from the max."""
    m = max(pos_prob, neg_prob)
    if m > cfg.impact_high_thr:
        return "HIGH"
    if m > cfg.impact_med_thr:
        return "MED"
    if m > cfg.impact_low_thr:
        return "LOW"
    return "NONE"


def apply_weights(df: pd.DataFrame, cfg: Config, today: date) -> pd.DataFrame:
    """Given scored articles + a Config, compute impact buckets, recency/trust/impact
    weights, and the cut flags. Pure function of df + cfg (no model calls)."""
    impact_w = {"HIGH": cfg.impact_w_high, "MED": cfg.impact_w_med,
                "LOW": cfg.impact_w_low, "NONE": 0.0}
    out = df.copy()
    out["impact"] = [impact_label(p, n, cfg) for p, n in zip(out["pos_prob"], out["neg_prob"])]

    out["rw_sent"] = out["date"].apply(
        lambda d: recency_weight(str(d)[:10], cfg.sent_half_life, today, cfg.recency_max_age))
    out["rw_news"] = out["date"].apply(
        lambda d: recency_weight(str(d)[:10], cfg.news_half_life, today, cfg.recency_max_age))

    out["sw_sent"] = out["source"].apply(lambda s: source_trust(s, cfg))
    out["sw_news"] = out["sw_sent"].apply(lambda s: s if s >= cfg.news_trust_gate else 0.0)

    iw = out["impact"].map(impact_w)
    out["w_sent"] = out["rw_sent"] * out["sw_sent"] * iw
    out["w_news"] = out["rw_news"] * out["sw_news"] * iw

    out["cut_sent"] = out["w_sent"] < cfg.cut_threshold
    out["cut_news"] = out["w_news"] < cfg.cut_threshold
    return out


def _weighted_inputs(df, tk, wcol, score_col, cut_col, sector_damp, macro_damp):
    """Return (values, weights) for a ticker after cut-filtering and applying the
    scope multiplier. Empty arrays if nothing survives."""
    sub = df[~df[cut_col]]
    scope = sub["scope"].values
    mult = np.zeros(len(sub))
    mult[scope == tk] = 1.0
    sec = SECTOR_OF.get(tk)
    if sec is not None:
        mult[scope == sec] = sector_damp
    mult[scope == "MACRO"] = macro_damp
    keep = mult > 0.0
    vals = sub[score_col].values[keep]
    w = sub[wcol].values[keep] * mult[keep]
    keep2 = w > 0.0
    return vals[keep2], w[keep2]


def _weighted_mean(vals, w):
    if len(vals) == 0 or w.sum() == 0:
        return 0.0
    return float((vals * w).sum() / w.sum())


def _bootstrap_ci(vals, w, rng, n_boot, lo=2.5, hi=97.5):
    """Percentile bootstrap CI for the weighted mean. Resamples (value, weight)
    pairs with replacement. Returns (lo, hi); degenerate cases collapse to the
    point estimate so the band is honestly zero-width, not fake-wide."""
    n = len(vals)
    if n == 0:
        return np.nan, np.nan
    if n == 1 or n_boot <= 0:
        m = _weighted_mean(vals, w)
        return m, m
    idx = rng.integers(0, n, size=(n_boot, n))
    v = vals[idx]
    ww = w[idx]
    wsum = ww.sum(axis=1)
    means = np.divide((v * ww).sum(axis=1), wsum,
                      out=np.zeros(n_boot), where=wsum > 0)
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def aggregate(df: pd.DataFrame, cfg: Config, today: date,
              tickers=TICKERS, rng=None) -> list[dict]:
    """Per-ticker aggregates with bootstrap CI bands + low-n flags.
    `df` must already be scored AND weighted (apply_weights)."""
    if rng is None:
        rng = np.random.default_rng(0)   # deterministic CI bands across runs
    rows = []
    for tk in tickers:
        sv, sw = _weighted_inputs(df, tk, "w_sent", "sent", "cut_sent",
                                  cfg.sent_sector_damp, cfg.sent_macro_damp)
        nv, nw = _weighted_inputs(df, tk, "w_news", "news", "cut_news",
                                  cfg.news_sector_damp, cfg.news_macro_damp)
        agg_sent, n_sent = _weighted_mean(sv, sw), len(sv)
        agg_news, n_news = _weighted_mean(nv, nw), len(nv)
        s_lo, s_hi = _bootstrap_ci(sv, sw, rng, cfg.bootstrap_n)
        n_lo, n_hi = _bootstrap_ci(nv, nw, rng, cfg.bootstrap_n)
        low_n = (n_sent < cfg.low_n_threshold) or (n_news < cfg.low_n_threshold)
        rows.append({
            "date": today.isoformat(), "ticker": tk,
            "agg_sent": round(agg_sent, 4), "agg_news": round(agg_news, 4),
            "n_sent": n_sent, "n_news": n_news,
            "sent_lo": round(s_lo, 4), "sent_hi": round(s_hi, 4),
            "news_lo": round(n_lo, 4), "news_hi": round(n_hi, 4),
            "low_n": bool(low_n),
        })
    return rows
