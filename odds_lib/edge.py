"""Deliverable 1 — per-class/subtype edge measurement.

Estimates the Brier-optimal edge multiplier k for each source class: how far
the model's deviation from the PRE-LOCK field proxy (c_hat) points toward
truth. Fit on c_hat (the predicted/shadow field value available pre-lock),
NEVER on the realized crowd (which is post-lock, diagnostic only).

  d        = p_model - c_hat
  residual = y - c_hat
  k_hat    = sum(d*resid) / sum(d^2)                 # UNCLIPPED diagnostic; may be < 0
  k_shrunk = (sum(d*resid) + lam*k_prior) / (sum(d^2) + lam)
  lam      = m_prior * d_bar_sq                       # squared-deviation units, NOT a row count
  deployed = clip(k_shrunk, 0, 1)                     # only the DEPLOYED k is clipped

A negative k_hat is a five-alarm signal: the class is anti-predictive (truth
moves opposite the model's deviation). We surface it; we don't deploy it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Structural priors by (source_class, subtype). Independent sharp sources get
# high conviction; odds-independent fallback gets none.
K_PRIOR = {
    ("MARKET", "market"): 0.90,
    ("ENGINE", "engine"): 0.90,
    ("PROP", "confirmed"): 0.75,
    ("PROP", "thin"): 0.40,
    ("RATE_SOT", "comparison"): 0.60,   # directionally robust (constant ~cancels)
    ("RATE_SOT", "single"): 0.50,
    ("RATE_SOT", "single_2h"): 0.50,
    # total_2h: c_hat is now the computed contest base rate (~0.62), not the
    # 0.49 global fallback — see field_model.TYPE_BASE_RATE. D5 slope/dispersion
    # audit (scripts/audit_total_sot_2h_slope.py) RESOLVED the open question:
    #   - the MEAN slope is data-backed (team OLS 3.03; full-match model mean
    #     10.34 vs observed 10.78) -> the recenter keeps it; do NOT flatten it.
    #   - the TAIL is overdispersed (full-match SOT var/mean=2.57 vs Poisson 1.0,
    #     not explainable by lambda spread) -> the single-mu Poisson P(4+) is
    #     OVER-confident at the extremes. k<1 (shrink toward c_hat) is the
    #     directionally-correct first-order fix for THAT, at both extremes.
    # So 0.50 is an EXPLICIT tail-overdispersion discount (NOT timid tempering of
    # a trusted tilt): recenter moves mu (mean, trusted), k shrinks tail curvature
    # (over-steep). Magnitude isn't calibratable (no paired SOT-lambda; n=9) -> a
    # neg-binomial tail is the eventual structural fix; until then 0.50 + the
    # n_active<4 freeze. (Was 0.90 when k routed around the bad 0.49 baseline.)
    ("RATE_SOT", "total_2h"): 0.50,
    ("SHADOW", "shadow"): 0.00,
}
M_PRIOR = 8.0            # pseudo-matches of prior conviction (prior dominates at small n)
D_BAR_SQ_FALLBACK = 0.04  # a typical squared deviation (0.20^2) when none observed
ACTIVE_D = 0.05          # |d| above which the model "took a position"
ACTIVE_FREEZE_N = 4      # below this many active rows, deployed k = k_prior (no drift)
MIN_CLUSTERS = 5         # below this many MATCH clusters, also freeze (correlated rows)


EDGE_CLIP_LO, EDGE_CLIP_HI = 0.02, 0.98  # final p_submit bounds


def edge_submit(p_model: float | None, c_hat: float | None, k: float) -> float:
    """The edge-weighted submission: ``c_hat + k*(p_model - c_hat)``, clipped.

    This is the SINGLE submission rule for every row. There is no other
    shrinkage path — all deviation from the field proxy is governed by k:
      * trusted class (high k)  -> lands near the raw model when it disagrees.
      * no-edge / SHADOW (k=0)  -> lands ON c_hat (no manufactured deviation).
      * no model (p_model None) -> lands ON c_hat (nothing to express).
    c_hat is the PRE-LOCK field proxy; never the realized (post-lock) crowd.
    """
    if c_hat is None:
        raise ValueError("edge_submit needs a c_hat (pre-lock field proxy)")
    c = float(c_hat)
    if p_model is None or k == 0.0:
        q = c
    else:
        q = c + float(k) * (float(p_model) - c)
    return min(max(q, EDGE_CLIP_LO), EDGE_CLIP_HI)


def deployed_k(cls: str, sub: str, table: pd.DataFrame | None = None) -> float:
    """Deployed k for a (class, subtype).

    Uses the fitted edge table's ``k_deployed`` (which already shrinks toward,
    and FREEZES on, the structural prior at small samples) when that class has
    resolved rows; otherwise the structural prior. At the current sample every
    class is frozen, so this returns the prior — by design.
    """
    if table is not None and not table.empty and (cls, sub) in table.index:
        return float(table.loc[(cls, sub), "k_deployed"])
    return float(K_PRIOR.get((cls, sub), 0.0))


def classify(tier: str, question_type: str) -> tuple[str, str]:
    t = (tier or "").strip()
    qt = (question_type or "").strip().lower()
    if t == "MARKET":
        return ("MARKET", "market")
    if t == "ENGINE_GOALS":
        return ("ENGINE", "engine")
    if t == "PROP_ok":
        return ("PROP", "confirmed")
    if t == "PROP_thin":
        return ("PROP", "thin")
    if t == "RATE_SOT_CMP":
        return ("RATE_SOT", "comparison")
    if t == "RATE_SOT":
        if "total_sot_2h" in qt:
            return ("RATE_SOT", "total_2h")
        if "2h" in qt:
            return ("RATE_SOT", "single_2h")
        return ("RATE_SOT", "single")
    return ("SHADOW", "shadow")


def _agg(g: pd.DataFrame, k_prior: float) -> pd.Series:
    d = (g["p_model"] - g["c_hat"]).to_numpy(dtype=float)
    resid = (g["y"] - g["c_hat"]).to_numpy(dtype=float)
    sd2 = float(np.sum(d ** 2))
    sd4 = float(np.sum(d ** 4))
    sdr = float(np.sum(d * resid))
    active = np.abs(d) > ACTIVE_D
    eff_n = (sd2 ** 2 / sd4) if sd4 > 0 else 0.0
    d_bar_sq = float(np.median(d[active] ** 2)) if active.any() else D_BAR_SQ_FALLBACK
    lam = M_PRIOR * d_bar_sq
    k_hat = (sdr / sd2) if sd2 > 0 else float("nan")
    k_shrunk = (sdr + lam * k_prior) / (sd2 + lam)
    n_active = int(active.sum())
    n_clusters = int(g["match"].nunique())
    # FREEZE: thin classes sit ON the prior, they don't drift on noise. Two
    # gates, BOTH required to unfreeze, because questions within a match share
    # one game script (correlated): enough active rows AND enough independent
    # MATCH clusters. Without the cluster gate a class with 6 active rows across
    # 3 lucky matches would crank k to its clipped max on ~3 correlated obs —
    # the exact "3 matches look good -> max k" trap. Frozen <=> LOW(prior-
    # dominated), so a prior-dominated class always deploys its structural prior.
    frozen = (n_active < ACTIVE_FREEZE_N) or (n_clusters < MIN_CLUSTERS)
    k_deployed = k_prior if frozen else min(max(k_shrunk, 0.0), 1.0)
    return pd.Series({
        "n": len(g),
        "clusters": g["match"].nunique(),
        "n_active": n_active,                # rows where the model took a position
        "eff_n_k": round(eff_n, 2),          # effective fitting sample for k
        "sum_d2": round(sd2, 4),
        "k_prior": k_prior,
        "k_hat": round(k_hat, 3) if sd2 > 0 else float("nan"),  # UNCLIPPED diagnostic
        "k_shrunk": round(k_shrunk, 3),                          # diagnostic
        "frozen": frozen,
        "k_deployed": round(min(max(k_deployed, 0.0), 1.0), 3),  # prior if frozen, else clipped shrunk
        "mean_rbp_final": round(g["rbp_final"].mean(), 2),
        "mean_rbp_model": round(g["rbp_model_cf"].dropna().mean(), 2) if g["rbp_model_cf"].notna().any() else float("nan"),
        "mean_rbp_base": round(g["rbp_baseline_cf"].dropna().mean(), 2) if g["rbp_baseline_cf"].notna().any() else float("nan"),
    })


def compute_edge_table(df: pd.DataFrame) -> pd.DataFrame:
    """df columns required: source_class, source_subtype, match, p_model, c_hat,
    y, rbp_final, rbp_model_cf, rbp_baseline_cf. Returns one row per
    (class, subtype) with the k estimators and confidence flags."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (cls, sub), g in df.groupby(["source_class", "source_subtype"]):
        kp = K_PRIOR.get((cls, sub), 0.0)
        s = _agg(g, kp)
        s["source_class"], s["source_subtype"] = cls, sub
        # confidence flag from match clustering + active fitting sample. The LOW
        # condition is EXACTLY the freeze condition, so frozen <=> prior-dominated.
        if s["clusters"] < MIN_CLUSTERS or s["n_active"] < ACTIVE_FREEZE_N:
            s["confidence"] = "LOW(prior-dominated)"
        elif s["clusters"] < 10:
            s["confidence"] = "MED"
        else:
            s["confidence"] = "OK"
        rows.append(s)
    rep = pd.DataFrame(rows).set_index(["source_class", "source_subtype"])
    cols = ["n", "clusters", "n_active", "eff_n_k", "sum_d2", "k_prior",
            "k_hat", "k_shrunk", "frozen", "k_deployed", "confidence",
            "mean_rbp_final", "mean_rbp_model", "mean_rbp_base"]
    return rep[cols].sort_index()


__all__ = ["classify", "compute_edge_table", "edge_submit", "deployed_k",
           "K_PRIOR", "M_PRIOR", "D_BAR_SQ_FALLBACK", "ACTIVE_D",
           "ACTIVE_FREEZE_N", "MIN_CLUSTERS", "EDGE_CLIP_LO", "EDGE_CLIP_HI"]
