"""Guards for the edge-weighted submission rule (Deliverable 3) + field anchor.

    .venv/bin/python tests/test_optimizer.py

Invariants: every row is p_submit = c_hat + k*(p_model - c_hat), clipped to
[0.02, 0.98]; trusted classes (high prior k) land NEAR the raw model; no-edge
rows (k=0 or no model) land ON c_hat (no manufactured deviation); k comes from
the structural prior when no fitted table is supplied; total_2h flows through
the SAME rule with c_hat=0.623 and is not double-pulled.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib.optimizer import optimize
from odds_lib.edge import edge_submit, deployed_k, K_PRIOR, classify
from odds_lib.field_model import FieldMeanEstimator


def _approx(a, b, tol=1e-9):
    return abs(a - b) < tol


# --- edge_submit primitive --------------------------------------------------

def test_edge_submit_blends_toward_model_by_k():
    # p_submit = c_hat + k*(p_model - c_hat)
    assert _approx(edge_submit(0.90, 0.50, 0.90), 0.50 + 0.90 * (0.90 - 0.50))


def test_edge_submit_k_zero_is_c_hat():
    assert _approx(edge_submit(0.90, 0.42, 0.0), 0.42)  # no manufactured deviation


def test_edge_submit_none_model_is_c_hat():
    assert _approx(edge_submit(None, 0.389, 0.90), 0.389)


def test_edge_submit_clips_to_band():
    assert edge_submit(0.999, 0.95, 1.0) <= 0.98
    assert edge_submit(0.001, 0.05, 1.0) >= 0.02


# --- deployed_k -------------------------------------------------------------

def test_deployed_k_falls_back_to_prior():
    assert _approx(deployed_k("MARKET", "market"), K_PRIOR[("MARKET", "market")])
    assert _approx(deployed_k("SHADOW", "shadow"), 0.0)


def test_deployed_k_uses_table_when_present():
    table = pd.DataFrame(
        {"k_deployed": [0.73]},
        index=pd.MultiIndex.from_tuples([("MARKET", "market")],
                                        names=["source_class", "source_subtype"]),
    )
    assert _approx(deployed_k("MARKET", "market", table), 0.73)
    # class absent from table -> prior
    assert _approx(deployed_k("ENGINE", "engine", table), K_PRIOR[("ENGINE", "engine")])


# --- optimize() end-to-end --------------------------------------------------

def test_market_lands_near_raw_model():
    s = optimize(tier="MARKET", p_hat=0.89, shadow=0.50)
    k = K_PRIOR[("MARKET", "market")]
    assert s.mode == "edge" and _approx(s.q, 0.50 + k * (0.89 - 0.50))
    # "near raw": within (1-k) of the model deviation, on the model's side of c_hat
    assert 0.50 < s.q < 0.89


def test_engine_high_prior_expresses_edge():
    s = optimize(tier="ENGINE_GOALS", p_hat=0.30, shadow=0.50)
    assert s.source_class == "ENGINE" and s.k == K_PRIOR[("ENGINE", "engine")]
    assert s.q < 0.50  # expressed toward the model, not stuck at the field


def test_no_model_shadows_c_hat():
    s = optimize(tier="PENDING", p_hat=None, shadow=0.389)
    assert s.mode == "shadow" and _approx(s.q, 0.389)


def test_unknown_tier_is_shadow_k_zero():
    s = optimize(tier="WHATEVER", p_hat=0.7, shadow=0.4)
    assert s.source_class == "SHADOW" and s.k == 0.0
    assert s.mode == "shadow" and _approx(s.q, 0.4)  # no deviation manufactured


def test_thin_prop_shrinks_more_than_confirmed():
    thin = optimize(tier="PROP_thin", p_hat=0.80, shadow=0.50)
    ok = optimize(tier="PROP_ok", p_hat=0.80, shadow=0.50)
    # both express edge, but the thin class (lower prior k) lands closer to c_hat
    assert 0.50 < thin.q < ok.q < 0.80


def test_sot_comparison_trusted_edge():
    s = optimize(tier="RATE_SOT_CMP", p_hat=0.194, shadow=0.542)
    assert s.source_subtype == "comparison" and s.mode == "edge"
    assert s.q < 0.542  # expresses the model's low read, not the role-blind field


def test_total_2h_routes_through_same_rule_no_double_pull():
    # c_hat=0.623 (type base rate). At mid-tempo the recentered model ~0.647,
    # so the edge pull is tiny -> lands between the two, NOT pulled twice to a
    # lower level. k is the total_2h prior (0.50).
    c_hat = 0.623
    s = optimize(tier="RATE_SOT", question_type="total_sot_2h_over",
                 p_hat=0.647, shadow=c_hat)
    assert s.source_subtype == "total_2h" and _approx(s.k, K_PRIOR[("RATE_SOT", "total_2h")])
    assert _approx(s.q, edge_submit(0.647, c_hat, 0.50))
    assert min(c_hat, 0.647) - 1e-9 <= s.q <= max(c_hat, 0.647) + 1e-9  # no overshoot


def test_total_2h_cagey_extreme_shrinks_toward_base_rate():
    # cagey model read 0.26 (most confident) shrinks toward c_hat=0.623 by (1-k):
    # the overdispersion discount, directionally correct (plug-in tail too steep).
    c_hat = 0.623
    s = optimize(tier="RATE_SOT", question_type="total_sot_2h_over",
                 p_hat=0.26, shadow=c_hat)
    assert 0.26 < s.q < c_hat  # pulled up from the over-steep tail, not all the way


def test_lower_bound_clamp_raises_below_floor():
    # goal-or-assist: blend (0.325+0.75*(0.57-0.325)=0.509) lands below the
    # anytime-goal lower bound 0.57 -> clamp up to the floor.
    s = optimize(tier="PROP_ok", question_type="player_goal_or_assist",
                 p_hat=0.57, shadow=0.325, k=0.75, lower_bound=True)
    assert s.lower_bound_clamped and _approx(s.q, 0.57)


def test_lower_bound_no_clamp_when_above_floor():
    s = optimize(tier="PROP_ok", question_type="player_goal_or_assist",
                 p_hat=0.40, shadow=0.60, k=0.75, lower_bound=True)
    assert not s.lower_bound_clamped and _approx(s.q, 0.45)   # 0.60+0.75*(0.40-0.60)


def test_lower_bound_never_clamps_shadow_row():
    s = optimize(tier="PENDING", question_type="player_goal_or_assist",
                 p_hat=None, shadow=0.325, lower_bound=True)
    assert not s.lower_bound_clamped and _approx(s.q, 0.325)


def test_lower_bound_off_by_default_does_not_clamp():
    # direct market (not a lower bound) below p_hat is NOT clamped — stays blended.
    s = optimize(tier="PROP_ok", question_type="player_goal", p_hat=0.57, shadow=0.325, k=0.75)
    assert not s.lower_bound_clamped and _approx(s.q, 0.325 + 0.75 * (0.57 - 0.325))


def test_table_overrides_prior_in_optimize():
    table = pd.DataFrame(
        {"k_deployed": [0.10]},
        index=pd.MultiIndex.from_tuples([("MARKET", "market")],
                                        names=["source_class", "source_subtype"]),
    )
    s = optimize(tier="MARKET", p_hat=0.90, shadow=0.50, table=table)
    assert _approx(s.k, 0.10) and _approx(s.q, 0.50 + 0.10 * (0.90 - 0.50))


# --- field estimator (unchanged) -------------------------------------------

def test_field_estimator_known_vs_unknown():
    fe = FieldMeanEstimator()
    known = fe.estimate("penalty_or_red_card")
    assert known.source == "qt_mean" and 0.3 < known.q_hat < 0.5
    unknown = fe.estimate("totally_made_up_type")
    assert unknown.source == "global_mean"


def test_field_estimator_type_base_rate_beats_global_fallback():
    fe = FieldMeanEstimator()
    est = fe.estimate("total_sot_2h_over")
    assert est.source == "type_base_rate"
    assert 0.58 < est.q_hat < 0.66
    assert abs(est.q_hat - fe.global_mean) > 0.1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
