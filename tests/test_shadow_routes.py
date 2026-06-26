"""Guards for the newly-founded shadow families:
  - both_teams_sot_1h  (closed-form volume model, engine lambda; OOS-gated; base fallback)
  - team_first_goal_2h (closed-form race of 2H Poissons; measured anchor fallback)
  - corner line-gap    (Poisson-ladder fit + measured base rate) -- see test_corners.py
Every route must be FOUNDED (varies per match) with a MEASURED, non-0.50 fallback.

    .venv/bin/python tests/test_shadow_routes.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib import match_engine as E, rate_layer as R, shadow_routes as SR, slate
from odds_lib.edge import classify, K_PRIOR, edge_submit
from odds_lib.optimizer import optimize


def _model(p_home, p_over=0.5, h1=0.44):
    return E.calibrate("A", "B", p_home, p_over, h1_share=h1)


# --- first_goal_2h: closed form -------------------------------------------------

def test_first_goal_2h_sums_to_p_any_2h_goal():
    m = _model(0.45)
    pa = E.p_team_first_goal_2h(m, "A")
    pb = E.p_team_first_goal_2h(m, "B")
    lam2 = (m.lam_home + m.lam_away) * (1 - m.h1_share)
    p_any = 1 - math.exp(-lam2)
    assert abs((pa + pb) - p_any) < 1e-9   # exactly partitions P(>=1 2H goal)

def test_first_goal_2h_tilts_to_favorite():
    m = _model(0.80)            # A heavy favorite -> lam_home >> lam_away
    assert E.p_team_first_goal_2h(m, "A") > E.p_team_first_goal_2h(m, "B")

def test_first_goal_2h_even_match_roughly_half_each():
    m = _model(0.40, p_over=0.5)   # near-even
    pa, pb = E.p_team_first_goal_2h(m, "A"), E.p_team_first_goal_2h(m, "B")
    assert abs(pa - pb) < 0.06     # symmetric-ish split

def test_first_goal_2h_route_and_k():
    m = _model(0.6)
    tier, p, _ = slate.resolve_row(
        {"question_type": "team_first_goal_2h", "target_team": "A", "line": ""},
        None, {"home_team": "A", "away_team": "B"}, (m, None, "A", "B", "market"))
    assert tier == "FIRST_GOAL_2H" and 0.0 < p < 1.0
    assert classify(tier, "x") == ("FIRST_GOAL_2H", "model") and K_PRIOR[("FIRST_GOAL_2H", "model")] == 1.0

def test_first_goal_2h_no_engine_uses_measured_anchor_not_half():
    tier, p, _ = slate.resolve_row(
        {"question_type": "team_first_goal_2h", "target_team": "A", "line": ""},
        None, {"home_team": "A", "away_team": "B"}, None)
    assert tier == "FIRST_GOAL_2H_BASE" and p == SR.first_goal_2h_anchor(True) and p != 0.50


# --- both_teams_sot_1h: closed form + OOS gate verdict --------------------------

def test_both_sot_1h_product_form():
    m = _model(0.5)
    rr = R.p_both_teams_sot_1h(m.lam_home, m.lam_away, m.h1_share)
    mu_h = R.team_sot_mu(m.lam_home, share=m.h1_share)
    mu_a = R.team_sot_mu(m.lam_away, share=m.h1_share)
    assert abs(rr.p - (1 - math.exp(-mu_h)) * (1 - math.exp(-mu_a))) < 1e-9

def test_both_sot_1h_varies_with_volume():
    lo = R.p_both_teams_sot_1h(*[x for x in (_model(0.5, 0.35).lam_home, _model(0.5, 0.35).lam_away)], 0.44)
    hi = R.p_both_teams_sot_1h(*[x for x in (_model(0.5, 0.75).lam_home, _model(0.5, 0.75).lam_away)], 0.44)
    assert hi.p > lo.p   # higher-scoring (volume) match -> both more likely to get a 1H SOT

def test_both_sot_1h_route_validated_uses_model():
    assert SR.both_sot_1h_validated()   # gate passed (scripts/fit_shadow_routes)
    m = _model(0.55, 0.6)
    tier, p, _ = slate.resolve_row(
        {"question_type": "both_teams_sot_1h", "target_team": "", "line": "0.5"},
        None, {"home_team": "A", "away_team": "B"}, (m, None, "A", "B", "market"))
    assert tier == "BOTH_SOT_1H" and 0.02 < p < 0.98
    assert classify(tier, "x") == ("BOTH_SOT_1H", "model") and K_PRIOR[("BOTH_SOT_1H", "model")] == 1.0

def test_both_sot_1h_no_engine_uses_measured_base_not_half():
    tier, p, _ = slate.resolve_row(
        {"question_type": "both_teams_sot_1h", "target_team": "", "line": "0.5"},
        None, {"home_team": "A", "away_team": "B"}, None)
    assert tier == "BOTH_SOT_1H_BASE" and p == round(SR.both_sot_1h_base_rate(), 4) and p != 0.50
    assert K_PRIOR[("BOTH_SOT_1H", "base")] == 1.0


# --- both_teams_sot_2h_1plus: RAW true P, k=1, NO anchor -----------------------

def test_both_sot_2h_product_form_2h_share():
    m = _model(0.5)
    rr = R.p_both_teams_sot_2h_1plus(m.lam_home, m.lam_away, m.h1_share)
    mu_h = R.team_sot_mu(m.lam_home, share=1 - m.h1_share)
    mu_a = R.team_sot_mu(m.lam_away, share=1 - m.h1_share)
    assert abs(rr.p - (1 - math.exp(-mu_h)) * (1 - math.exp(-mu_a))) < 1e-9

def test_both_sot_2h_ships_raw_model_undistorted():
    # k=1 and NO anchor: the submitted value must EXACTLY equal the raw closed form
    # (no blending toward the measured base rate).
    m = _model(0.55, 0.6)
    tier, p, _ = slate.resolve_row(
        {"question_type": "both_teams_sot_2h_1plus", "target_team": "", "line": "0.5"},
        None, {"home_team": "A", "away_team": "B"}, (m, None, "A", "B", "market"))
    assert tier == "BOTH_SOT_2H"
    assert classify(tier, "x") == ("BOTH_SOT_2H", "model") and K_PRIOR[("BOTH_SOT_2H", "model")] == 1.0
    raw = R.p_both_teams_sot_2h_1plus(m.lam_home, m.lam_away, m.h1_share).p
    sub = optimize(tier=tier, question_type="both_teams_sot_2h_1plus", p_hat=p, shadow=SR.both_sot_2h_base_rate())
    assert abs(sub.q - round(raw, 4)) < 1e-9   # submitted == raw model; NOT pulled to base 0.81

def test_both_sot_2h_no_engine_uses_measured_base():
    tier, p, _ = slate.resolve_row(
        {"question_type": "both_teams_sot_2h_1plus", "target_team": "", "line": "0.5"},
        None, {"home_team": "A", "away_team": "B"}, None)
    assert tier == "BOTH_SOT_2H_BASE" and p == round(SR.both_sot_2h_base_rate(), 4) and p != 0.50


# --- offsides: MEASURED P(>=2) raw, home/away, never 0.50 ----------------------

def test_offsides_route_is_pooled_floor_no_edge():
    # NO per-match signal OOS -> honest pooled FLOOR; home & away get the SAME value (the
    # split is in-sample noise, not used). Measured, never 0.50; flagged no-edge.
    g = {"home_team": "A", "away_team": "B"}
    th, ph, _ = slate.resolve_row({"question_type": "team_offsides_over", "target_team": "A", "line": "1.5"}, None, g, None)
    ta, pa, _ = slate.resolve_row({"question_type": "team_offsides_over", "target_team": "B", "line": "1.5"}, None, g, None)
    assert th == "OFFSIDES_FLOOR" and ta == "OFFSIDES_FLOOR"
    # uncovered (A/B) -> the EB POOLED PRIOR (n=0 limit of the per-team model); pooled (home==away), not 0.50.
    assert ph == pa and ph != 0.50
    assert abs(ph - SR.offside_pooled_prior(1.5)) < 1e-9
    assert SR.offsides_is_floor_no_edge()                                # honestly flagged no-edge
    assert classify("OFFSIDES_FLOOR", "x") == ("OFFSIDES", "floor") and K_PRIOR[("OFFSIDES", "floor")] == 1.0


# --- 2H cards: MEASURED per-threshold floor (no clean driver OOS), never crowd-copy ---

def test_cards_2h_floor_team_and_total():
    g = {"home_team": "A", "away_team": "B"}
    # team_card_2h = team >=1 (line 0.5)
    t1, p1, _ = slate.resolve_row({"question_type": "team_card_2h", "target_team": "A", "line": ""}, None, g, None)
    # team_cards_2h_over at a line
    t2, p2, _ = slate.resolve_row({"question_type": "team_cards_2h_over", "target_team": "A", "line": "1.5"}, None, g, None)
    # total_cards_2h_over at a line
    t3, p3, _ = slate.resolve_row({"question_type": "total_cards_2h_over", "target_team": "", "line": "3.5"}, None, g, None)
    assert t1 == t2 == t3 == "CARDS_2H_FLOOR"
    assert p1 == round(SR.cards_2h_rate("team", 0.5), 4) and p1 != 0.50
    assert p2 == round(SR.cards_2h_rate("team", 1.5), 4)
    assert p3 == round(SR.cards_2h_rate("total", 3.5), 4) and p3 != 0.50
    assert p1 > p2                                  # P(>=1) > P(>=2) monotone CDF
    assert SR.cards_2h_is_floor()                   # honestly flagged: measured floor, no clean signal
    assert classify("CARDS_2H_FLOOR", "x") == ("CARDS_2H", "floor") and K_PRIOR[("CARDS_2H", "floor")] == 1.0


# --- penalty/red: SOURCED external anchors raw, never 0.50 ---------------------

def test_penalty_anchors_measured_not_half():
    g = {"home_team": "A", "away_team": "B"}
    t1, p1, _ = slate.resolve_row({"question_type": "penalty_awarded", "target_team": "", "line": ""}, None, g, None)
    t2, p2, _ = slate.resolve_row({"question_type": "penalty_or_red_card", "target_team": "", "line": ""}, None, g, None)
    assert t1 == "PENALTY_BASE" and t2 == "PENALTY_BASE"
    assert p1 == round(SR.penalty_anchor("penalty_awarded"), 4) and p1 != 0.50
    assert p2 > p1 and p2 != 0.50                       # union >= penalty-only
    assert classify("PENALTY_BASE", "x") == ("PENALTY", "base") and K_PRIOR[("PENALTY", "base")] == 1.0


# --- benched-player minutes-scaled sub prop (Item 4) ---------------------------

def test_benched_prop_minutes_scaled_not_shadow():
    from odds_lib.player_prop_pricing import minutes_scaled_sub
    # a benched STAR (starter read 0.357, Balogun-like) -> founded minutes-scaled value, NOT c_hat
    p = minutes_scaled_sub(0.357, "bench_high_usage")
    assert 0.08 < p < 0.20 and p < 0.357          # scaled below the starter read, not zero
    # usage ordering: high-usage sub > low-usage sub
    assert minutes_scaled_sub(0.357, "bench_high_usage") > minutes_scaled_sub(0.357, "bench_low_usage")
    # out_of_squad / unknown -> None (route stays PENDING, not a constant)
    assert minutes_scaled_sub(0.357, "out_of_squad") is None
    assert classify("PROP_SUB", "player_goal") == ("PROP", "sub") and K_PRIOR[("PROP", "sub")] == 1.0


# --- measured anchors are real corpus rates, never 0.50 ------------------------

def test_measured_fallbacks_are_not_half():
    assert SR.both_sot_1h_base_rate() not in (None, 0.5)
    assert SR.corner_base_rate("team", 4.5) not in (None, 0.5)
    assert SR.corner_base_rate("total", 8.5) not in (None, 0.5)
    assert SR.first_goal_2h_anchor(True) != SR.first_goal_2h_anchor(False)  # home/away differ


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
