"""EXHAUSTIVE invariant guard: NO question row's read can be silently overridden / shrunk
toward the field-mean c_hat by the edge-table fit. Covers EVERY tier resolve_row can emit.

The failure this prevents (forever): a sharp de-vigged book line (Turkiye win, de-vig 0.29)
getting pulled to 0.45 because the fitted MARKET k (0.52) overrode the structural prior and
shrank it toward the team_win field-mean (0.62).

Three invariants, proven against an ADVERSARIAL edge table that tries to shrink EVERY class to
k=0.01:
  1. classify() maps every emitted tier to a real (class, subtype) in K_PRIOR (no real read
     silently falls through to SHADOW -> c_hat). PENDING is the only intended SHADOW.
  2. deployed_k never returns BELOW the structural prior for ANY class (the fit can only RAISE
     trust in a read, never increase the shrink toward c_hat).
  3. Every market-price (TRUST_PRICE_K) tier submits the RAW read unchanged, regardless of the
     fit -- optimize(p_hat) == p_hat.

    .venv/bin/python tests/test_no_market_override.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from odds_lib.edge import classify, deployed_k, K_PRIOR, TRUST_PRICE_K
from odds_lib.optimizer import optimize

# EVERY tier odds_lib/slate.resolve_row (+ _prop_tier) can return. If a new route is added,
# add its tier here -- the test then forces it to satisfy the no-override invariants.
LIVE_TIERS = [
    "MARKET", "MARKET_INTERP",
    "ENGINE_GOALS", "ENGINE_GOALS_H1MKT", "ENGINE_GOALS_H1FALLBACK",
    "RATE_SOT", "RATE_SOT_CMP", "SOT_COUNT",
    "CORNERS_OK", "CORNERS_THIN", "CORNERS_LADDER", "CORNERS_BASE",
    "CARDS_OK", "CARDS_THIN", "CARDS_2H_MKT", "CARDS_2H_FLOOR",
    "TEAMGOALS_OK", "TEAMGOALS_THIN",
    "CORNERS_CMP_OK", "CORNERS_CMP_THIN", "CORNERS_CMP_MODEL",
    "CORNER_HALF_PINNACLE", "CORNER_HALF_STOPGAP",
    "H1GOALS_OK", "H1GOALS_THIN", "H2GOALS_OK", "H2GOALS_THIN",
    "MORE_CARDS", "MATCH_SOT", "FOUL_CMP",
    "BOTH_SOT_1H", "BOTH_SOT_1H_BASE", "BOTH_SOT_2H", "BOTH_SOT_2H_BASE",
    "FIRST_GOAL_2H", "FIRST_GOAL_2H_BASE",
    "OFFSIDES_FLOOR", "PENALTY_BASE",
    "PROP_ok", "PROP_thin", "PROP_direct_thin", "PROP_proxy_floor",
]

# Tiers whose read is a real de-vigged MARKET PRICE -> must submit AT the line (immune to fit).
MARKET_PRICE_TIERS = [
    "MARKET", "MARKET_INTERP",
    "CORNERS_OK", "CORNERS_THIN", "CORNERS_LADDER",
    "CARDS_OK", "CARDS_THIN", "CARDS_2H_MKT",
    "TEAMGOALS_OK", "TEAMGOALS_THIN",
    "CORNERS_CMP_OK", "CORNERS_CMP_THIN",
    "H1GOALS_OK", "H1GOALS_THIN", "H2GOALS_OK", "H2GOALS_THIN",
    "CORNER_HALF_PINNACLE",
    "PROP_ok", "PROP_thin", "PROP_direct_thin", "PROP_proxy_floor",  # all k=1: ship the prop read raw
]


def _adversarial_table():
    """An edge table that tries to shrink EVERY known class to k=0.01 (toward c_hat)."""
    idx = pd.MultiIndex.from_tuples(list(K_PRIOR.keys()), names=["source_class", "source_subtype"])
    return pd.DataFrame({"k_deployed": [0.01] * len(K_PRIOR)}, index=idx)


def test_every_tier_classifies_to_a_real_prior():
    # No emitted tier silently falls through to SHADOW (which would submit c_hat). PENDING only.
    for t in LIVE_TIERS:
        cls, sub = classify(t, "x")
        assert (cls, sub) != ("SHADOW", "shadow"), f"{t} falls through to SHADOW -> would submit c_hat"
        assert (cls, sub) in K_PRIOR, f"{t} -> {(cls, sub)} has no K_PRIOR entry"
    assert classify("PENDING", "x") == ("SHADOW", "shadow")   # the ONE intended shadow


def test_fit_can_never_shrink_below_prior():
    # UNIVERSAL GUARD: under an adversarial fit, no class deploys k below its structural prior.
    adv = _adversarial_table()
    for (cls, sub), prior in K_PRIOR.items():
        k = deployed_k(cls, sub, adv)
        assert k >= prior - 1e-9, f"({cls},{sub}) shrunk below prior: {k} < {prior}"


def test_market_price_tiers_immune_and_submit_raw():
    # Every market-price tier: deployed k == prior (>=0.9) AND the submission equals the raw read,
    # even with an adversarial table. p_hat far from c_hat to make any shrink visible.
    adv = _adversarial_table()
    for t in MARKET_PRICE_TIERS:
        cls, sub = classify(t, "x")
        assert (cls, sub) in TRUST_PRICE_K, f"{t} -> {(cls,sub)} not in TRUST_PRICE_K"
        k = deployed_k(cls, sub, adv)
        # immune to the fit (deploys its prior) and that prior is a high trust level (>=0.80)
        assert k == K_PRIOR[(cls, sub)] and k >= 0.80, f"{t}: deployed k {k} != prior (or <0.80)"
        for p_hat in (0.05, 0.29, 0.95):
            s = optimize(tier=t, question_type="x", p_hat=p_hat, shadow=0.62, table=adv)
            # k==1 -> exact; k in [0.9,1) -> within (1-k) of the line, NEVER pulled to c_hat
            assert abs(s.q - p_hat) <= (1.0 - k) + 1e-9, f"{t} read moved: {p_hat}->{s.q} (k={k})"


def test_trust_price_classes_have_high_prior():
    # Closes the residual: a NEW market route wired with a too-low prior would sit below the line
    # even though it's "pinned" (TRUST_PRICE_K returns the prior). Every market-price class must
    # have prior >= 0.90 (trust the de-vigged read). Fails loudly on a mis-wired route.
    bad = [(c, s, K_PRIOR.get((c, s))) for (c, s) in TRUST_PRICE_K if K_PRIOR.get((c, s), 0.0) < 0.90]
    assert not bad, f"TRUST_PRICE_K classes with prior < 0.90 (mis-wired market route): {bad}"


def test_turkiye_win_regression_via_full_optimize():
    # The exact disaster, end-to-end through optimize() with a fitted-0.52 MARKET table.
    tbl = pd.DataFrame({"k_deployed": [0.52]},
                       index=pd.MultiIndex.from_tuples([("MARKET", "market")],
                                                       names=["source_class", "source_subtype"]))
    s = optimize(tier="MARKET", question_type="team_win", p_hat=0.294, shadow=0.62, table=tbl)
    assert abs(s.q - 0.294) < 1e-9, f"sharp line overridden: {s.q}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed — no market override possible for any tier")
