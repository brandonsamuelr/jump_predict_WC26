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


def test_deployed_k_market_immune_to_table_fit():
    # HARD RULE: a sharp de-vigged book line (TRUST_PRICE_K) is NEVER shrunk by the edge-table
    # fit -> always the structural prior, even if the table carries a (wrong) shrink. This is
    # the guard against the Turkiye-win override (de-vig 0.29 pulled to 0.45 by a fitted 0.52).
    table = pd.DataFrame(
        {"k_deployed": [0.52]},
        index=pd.MultiIndex.from_tuples([("MARKET", "market")],
                                        names=["source_class", "source_subtype"]),
    )
    assert _approx(deployed_k("MARKET", "market", table), K_PRIOR[("MARKET", "market")])  # ignores 0.52


def test_universal_guard_raises_to_table_but_floors_at_prior():
    # The universal guard: deployed_k = min(max(prior, fitted), 1.0). The fit may RAISE k toward
    # a read, never lower it below the prior. (All LIVE classes are now prior=1.0, so this is
    # exercised via a synthetic class.) NOTE: as of 2026-06-26 every real model class is k=1, so
    # the fit is inert on the live path -- this guards the MECHANISM against future k<1 routes.
    from odds_lib import edge as E
    E.K_PRIOR[("_GUARDTEST", "x")] = 0.40
    try:
        hi = pd.DataFrame({"k_deployed": [0.85]}, index=pd.MultiIndex.from_tuples(
            [("_GUARDTEST", "x")], names=["source_class", "source_subtype"]))
        lo = pd.DataFrame({"k_deployed": [0.10]}, index=pd.MultiIndex.from_tuples(
            [("_GUARDTEST", "x")], names=["source_class", "source_subtype"]))
        assert _approx(deployed_k("_GUARDTEST", "x", hi), 0.85)   # fit ABOVE prior -> raised
        assert _approx(deployed_k("_GUARDTEST", "x", lo), 0.40)   # fit BELOW prior -> floored
    finally:
        del E.K_PRIOR[("_GUARDTEST", "x")]


def test_sharp_market_line_never_overridden_regression():
    # REGRESSION (Turkiye-win disaster): de-vig 0.294, field-mean c_hat 0.62, and an edge table
    # that fitted k=0.52 for MARKET. The submission MUST be the line (~0.294), NOT pulled to 0.45.
    table = pd.DataFrame(
        {"k_deployed": [0.52]},
        index=pd.MultiIndex.from_tuples([("MARKET", "market")],
                                        names=["source_class", "source_subtype"]),
    )
    s = optimize(tier="MARKET", p_hat=0.294, shadow=0.62, table=table)
    assert abs(s.q - 0.294) < 0.01, f"sharp line overridden: {s.q}"


# --- optimize() end-to-end --------------------------------------------------

def test_market_lands_at_raw_line():
    # MARKET k=1.0 (trust the de-vigged line) -> submit AT the line, never shrunk toward c_hat.
    s = optimize(tier="MARKET", p_hat=0.89, shadow=0.50)
    k = K_PRIOR[("MARKET", "market")]
    assert k == 1.0 and s.mode == "edge" and _approx(s.q, 0.89)   # lands ON the line


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


def test_all_confirmed_starter_props_ship_raw():
    # Item 1 (2026-06-26): every confirmed-starter prop read is a real de-vigged market -> k=1.
    # Thin = variance not bias; NO shrink toward c_hat. thin == ok == direct_thin == p_hat.
    for tier in ("PROP_ok", "PROP_thin", "PROP_direct_thin", "PROP_proxy_floor"):
        s = optimize(tier=tier, p_hat=0.80, shadow=0.50)
        assert s.k == 1.0 and _approx(s.q, 0.80), f"{tier} not shipping raw: {s.q}"


def test_sot_comparison_trusted_edge():
    s = optimize(tier="RATE_SOT_CMP", p_hat=0.194, shadow=0.542)
    assert s.source_subtype == "comparison" and s.mode == "edge"
    assert s.q < 0.542  # expresses the model's low read, not the role-blind field


def test_total_2h_ships_raw_at_k1_no_pull():
    # FIXED 2026-06-26: total_2h is k=1.0 (was 0.50). The read is submitted RAW, NOT pulled
    # toward c_hat=0.623. (Offset also removed in rate_layer; total 2H SOT ~Poisson.)
    s = optimize(tier="RATE_SOT", question_type="total_sot_2h_over", p_hat=0.647, shadow=0.623)
    assert s.source_subtype == "total_2h" and _approx(s.k, 1.0)
    assert _approx(s.q, 0.647)   # equals the read, independent of c_hat


def test_total_2h_ships_raw_no_pull_to_base_rate():
    # FIXED 2026-06-26: total_sot_2h_over is now k=1 (offset removed; total 2H SOT ~Poisson,
    # var/mean 1.21). The read is shipped RAW -- NOT shrunk toward the c_hat=0.623 placeholder.
    s = optimize(tier="RATE_SOT", question_type="total_sot_2h_over", p_hat=0.26, shadow=0.623)
    assert s.k == 1.0 and abs(s.q - 0.26) < 1e-9   # submits the read, independent of c_hat


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


# (the fit-raises/floors mechanism is now tested via the synthetic class in
#  test_universal_guard_raises_to_table_but_floors_at_prior — every LIVE class is k=1.)


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
