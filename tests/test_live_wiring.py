"""Guards for the two live-pricing changes (k anchor-quality fix + SOT count model).

    .venv/bin/python tests/test_live_wiring.py

Asserts (the entry-shift-lesson guardrails):
  1. k-fix is SURGICAL: a market-priced row now submits AT/NEAR the real price
     (not blended halfway toward the placeholder); a placeholder-shadow row is
     UNCHANGED (still submits c_hat exactly).
  2. SOT swap is correct: team_sot_over -> SOT_COUNT (new logistic); team_more_sot
     comparison path unchanged (RATE_SOT_CMP); both_teams_sot uses shadow (PENDING),
     NOT the SOT model.
  3. clip bounds [0.02,0.98] and the lower-bound clamp still hold.
  4. no feature leakage: the SOT model uses only pre-kickoff odds-derived features.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib.optimizer import optimize
from odds_lib.edge import (edge_submit, K_PRIOR, EDGE_CLIP_LO, EDGE_CLIP_HI,
                           GATE_MODEL_TRUST_K, classify)
from odds_lib import slate, match_engine as E, sot_count_model as SC


def _approx(a, b, t=1e-9):
    return abs(a - b) < t


# --- 1. k-fix surgical ------------------------------------------------------

def test_market_priced_row_lands_near_real_price_not_halfway():
    # MARKET: real de-vigged price 0.562 vs placeholder field-mean 0.474.
    s = optimize(tier="MARKET", p_hat=0.562, shadow=0.474)
    old_halfway = edge_submit(0.562, 0.474, 0.50)   # the diluted behavior we rejected
    assert s.q > old_halfway                         # no longer dragged halfway back
    assert abs(s.q - 0.562) < 0.20 * abs(0.562 - 0.474)  # within 20% of the price

def test_prop_direct_thin_no_longer_halfway_diluted():
    # the corners-lesson pattern: a real (thin) market price must not be blended
    # halfway toward the field-mean placeholder.
    s = optimize(tier="PROP_direct_thin", p_hat=0.449, shadow=0.325)
    assert s.q > edge_submit(0.449, 0.325, 0.50)     # beats the old k=0.50 dilution
    assert abs(s.q - 0.449) < abs(0.387 - 0.449)     # closer to the price than old 0.387

def test_placeholder_shadow_row_unchanged():
    # genuinely-hard placeholder shadow: k=0, submits c_hat EXACTLY (untouched).
    assert _approx(K_PRIOR[("SHADOW", "shadow")], 0.0)
    s = optimize(tier="PENDING", p_hat=None, shadow=0.49)   # PENDING -> SHADOW class
    assert s.source_class == "SHADOW" and s.mode == "shadow" and _approx(s.q, 0.49)


# --- 2. SOT swap correct ----------------------------------------------------

def _consensus():
    return pd.DataFrame([
        {"market_key": "h2h", "line": float("nan"), "outcome": "H", "market_prob": 0.50},
        {"market_key": "h2h", "line": float("nan"), "outcome": "Draw", "market_prob": 0.27},
        {"market_key": "h2h", "line": float("nan"), "outcome": "A", "market_prob": 0.23},
        {"market_key": "totals", "line": 2.5, "outcome": "Over", "market_prob": 0.55},
        {"market_key": "totals", "line": 2.5, "outcome": "Under", "market_prob": 0.45},
    ])

def _model_tuple():
    m = E.calibrate("H", "A", p_home=0.50, p_over=0.55)
    return (m, E.simulate(m, n=40_000, seed=3), "H", "A")

def test_team_sot_over_uses_new_logistic():
    c, mt = _consensus(), _model_tuple()
    row = {"question_type": "team_sot_over", "target_team": "H", "line": "2.5"}
    tier, p, _ = slate.resolve_row(row, c, {}, mt)
    expect = SC.predict_team_sot_over(0.50 - 0.23, 0.55, 2.5)   # fav_gap, total, line
    assert tier == "SOT_COUNT"
    assert classify(tier, "team_sot_over") == ("SOT_COUNT", "model")
    assert abs(p - round(expect, 4)) < 1e-3

def test_team_more_sot_comparison_path_unchanged():
    c, mt = _consensus(), _model_tuple()
    row = {"question_type": "team_more_sot_2h", "target_team": "H", "line": ""}
    tier, p, _ = slate.resolve_row(row, c, {}, mt)
    assert tier == "RATE_SOT_CMP"   # comparison path NOT rerouted

def test_both_teams_sot_uses_shadow_not_sot_model():
    c, mt = _consensus(), _model_tuple()
    row = {"question_type": "both_teams_sot_1plus", "target_team": "", "line": ""}
    tier, p, _ = slate.resolve_row(row, c, {}, mt)
    assert tier == "PENDING" and p is None   # shadow, not any SOT model


# --- 3. clip + lower-bound clamp --------------------------------------------

def test_clip_bounds_hold():
    assert edge_submit(0.999, 0.95, 1.0) <= EDGE_CLIP_HI
    assert edge_submit(0.001, 0.05, 1.0) >= EDGE_CLIP_LO

def test_lower_bound_clamp_holds():
    s = optimize(tier="PROP_proxy_floor", question_type="player_goal_or_assist",
                 p_hat=0.57, shadow=0.30, k=0.40, lower_bound=True)
    assert s.lower_bound_clamped and _approx(s.q, 0.57)


# --- 4. no feature leakage --------------------------------------------------

def test_sot_model_features_are_pre_kickoff_only():
    meta = SC._load()["_meta"]
    # ONLY odds-derived, pre-kickoff features (computable at lock from the Odds API)
    assert meta["features"] == ["favorite_gap", "total_line_prob"]
    # gate-validated model submits UNDISTORTED (k=1, no toward-shadow hedge)
    assert _approx(K_PRIOR[("SOT_COUNT", "model")], GATE_MODEL_TRUST_K)
    assert _approx(GATE_MODEL_TRUST_K, 1.0)


def test_sot_count_submits_undistorted():
    # team_sot_over now submits the model probability, not a toward-shadow blend
    s = optimize(tier="SOT_COUNT", question_type="team_sot_over", p_hat=0.42, shadow=0.49)
    assert _approx(s.q, 0.42)   # undistorted (was 0.49 + 0.5*(0.42-0.49) = 0.455)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
