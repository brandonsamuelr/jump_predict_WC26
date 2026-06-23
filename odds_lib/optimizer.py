"""Breadth-first submission optimizer — Deliverable 3, the edge-weighted rule.

Turns a per-question (model probability p_hat, confidence tier, field proxy
c_hat) into a final submission, on EVERY question. One rule, no exceptions:

    p_submit = c_hat + k * (p_model - c_hat)      [clipped to [0.02, 0.98]]

    - c_hat  = the PRE-LOCK field proxy for the row (the shadow / qt-mean /
               type base rate). NEVER the realized post-lock crowd.
    - p_model= our independent model probability (None on no-model rows).
    - k      = the per-(class, subtype) edge multiplier (edge.deployed_k):
               the fitted-and-frozen edge table value when the class has
               resolved rows, else the structural prior. At the current sample
               every class is frozen on its prior, by design.

Why this replaced trust-or-shadow + variance_tilt
-------------------------------------------------
The old optimizer submitted p_hat EXACTLY on trusted tiers (an implicit k=1)
and offered a blanket variance_tilt that overshot p_hat to "buy variance" —
EV-negative variance for its own sake. The strategy now is: express genuine
edge fully through k (MARKET/ENGINE priors are HIGH on purpose), and add
variance only where it comes from independently justified edge. So:
  * trusted class (high k)  -> lands NEAR the raw model when it disagrees
                               with the field (the edge is expressed).
  * no-edge / SHADOW (k=0)  -> lands ON c_hat (we don't manufacture deviation).
There is no silent shrinkage path: every departure from the model, and every
departure from the field, is the single k.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .edge import classify, deployed_k, edge_submit


@dataclass
class Submission:
    q: float
    mode: str               # "edge" (k>0, has model) | "shadow" (k=0 or no model)
    tier: str
    source_class: str
    source_subtype: str
    p_hat: float | None     # the raw model probability (p_model)
    shadow: float | None    # the pre-lock field proxy (c_hat)
    k: float                # the deployed edge multiplier
    note: str


def optimize(
    *,
    tier: str,
    p_hat: float | None,
    shadow: float | None,
    question_type: str = "",
    table: pd.DataFrame | None = None,
    k: float | None = None,
) -> Submission:
    """Compute the submission for one question via the edge-weighted rule.

    ``tier`` + ``question_type`` are mapped to a (class, subtype) and the
    deployed k looked up (from the fitted ``table`` if given, else the
    structural prior). Pass an explicit ``k`` to override (tests/diagnostics).
    ``shadow`` (c_hat) is required — it is where a no-edge row lands.
    """
    cls, sub = classify(tier, question_type)
    kk = deployed_k(cls, sub, table) if k is None else float(k)
    q = edge_submit(p_hat, shadow, kk)
    if p_hat is None or kk == 0.0:
        mode = "shadow"
        note = "submit c_hat (no edge)"
    else:
        mode = "edge"
        note = f"c_hat + {kk:.2f}*(p_model - c_hat)"
    return Submission(q=q, mode=mode, tier=tier, source_class=cls,
                      source_subtype=sub, p_hat=p_hat, shadow=shadow, k=kk,
                      note=note)


__all__ = ["Submission", "optimize"]
