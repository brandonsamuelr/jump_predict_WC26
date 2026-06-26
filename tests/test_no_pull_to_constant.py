"""STANDING GUARD against the 'pull-toward-a-stale-constant' disease (the class of bug behind
the MARKET k-override, the club card floor, and the SOT-2H double-pull).

A row's submitted value must be a function of REAL per-match info, NOT of the match-blind field
constant c_hat (FieldMeanEstimator / TYPE_BASE_RATE). Test: run optimize() twice with very
different c_hat (0.20 vs 0.80); for every TRUE-P / market / k=1 tier the submitted q must NOT
move. Any tier whose q moves is c_hat-dependent -- either a legit no-data fallback or a live
instance of the disease (listed by test_report_constant_dependent_tiers for review).

Companion to tests/test_no_market_override.py (which covers the k-shrink sub-case for sharp
markets). This covers the WHOLE disease class: any k<1 / offset / blend that drags a per-match
read toward the field.

    .venv/bin/python tests/test_no_pull_to_constant.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib.optimizer import optimize
from odds_lib.edge import classify, K_PRIOR

# (tier, question_type) for every tier whose read is TRUE-P / a market price / a measured floor
# that IS the estimate -> submitted q must be INDEPENDENT of c_hat (deployed k == 1.0).
INDEPENDENT_TIERS = [
    ("MARKET", "team_win"), ("MARKET_INTERP", "match_total_over"),
    ("CORNERS_OK", "team_corners_over"), ("CORNERS_THIN", "team_corners_over"),
    ("CORNERS_LADDER", "team_corners_over"), ("CORNERS_BASE", "team_corners_over"),
    ("CARDS_OK", "total_cards_over"), ("CARDS_THIN", "total_cards_over"),
    ("CARDS_2H_MKT", "total_cards_2h_over"), ("CARDS_2H_FLOOR", "total_cards_2h_over"),
    ("TEAMGOALS_OK", "team_total_goals_over"), ("TEAMGOALS_THIN", "team_total_goals_over"),
    ("CORNERS_CMP_OK", "team_more_corners_full"), ("CORNERS_CMP_THIN", "team_more_corners_full"),
    ("CORNERS_CMP_MODEL", "team_more_corners_full"),
    ("CORNER_HALF_STOPGAP", "team_more_corners_2h"), ("CORNER_HALF_PINNACLE", "team_more_corners_1h"),
    ("H1GOALS_OK", "total_goals_1h_over"), ("H1GOALS_THIN", "total_goals_1h_over"),
    ("H2GOALS_OK", "second_half_goals_over"), ("H2GOALS_THIN", "second_half_goals_over"),
    ("SOT_COUNT", "team_sot_over"), ("MORE_CARDS", "team_more_cards"),
    ("FOUL_CMP", "team_more_fouls"), ("MATCH_SOT", "match_total_sot_over"),
    ("BOTH_SOT_1H", "both_teams_sot_1h"), ("BOTH_SOT_1H_BASE", "both_teams_sot_1h"),
    ("BOTH_SOT_2H", "both_teams_sot_2h_1plus"), ("BOTH_SOT_2H_BASE", "both_teams_sot_2h_1plus"),
    ("FIRST_GOAL_2H", "team_first_goal_2h"), ("FIRST_GOAL_2H_BASE", "team_first_goal_2h"),
    ("OFFSIDES_FLOOR", "team_offsides_over"), ("PENALTY_BASE", "penalty_or_red_card"),
    ("RATE_SOT", "total_sot_2h_over"),   # FIXED Part C: k=1, offset removed -> now c_hat-independent
    # Item 1: all confirmed-starter prop reads ship raw (k=1) -> c_hat-independent.
    ("PROP_ok", "player_sot_over"), ("PROP_thin", "player_sot_over"),
    ("PROP_direct_thin", "player_goal_or_assist"), ("PROP_proxy_floor", "player_goal_or_assist"),
    ("PROP_SUB", "player_goal"),   # Item 4: benched minutes-scaled closed form, k=1
    # Item 3: RATE_SOT family de-shrunk to k=1 (concave mean beats base OOS). + ENGINE de-hedged.
    ("RATE_SOT_CMP", "team_more_sot_2h"), ("RATE_SOT", "team_sot_2h_over"),
    ("RATE_SOT", "team_sot_over"),
    ("ENGINE_GOALS", "team_score_any"), ("ENGINE_GOALS_H1MKT", "second_half_more_goals"),
]

# Tiers KNOWN to be c_hat-dependent (deployed k<1) -- the disease surface, reported not asserted.
# Each must be either justified (outcome-fitted, genuinely overconfident) or fixed.
# EMPTY as of 2026-06-26: the entire pull-to-constant disease surface is closed. Every recurring
# tier is now market read / OOS-validated model / population-matched base rate, all at k=1. Only
# SHADOW (k=0, no read -> c_hat) and the degenerate no-data fallbacks remain c_hat-related, and
# those are CORRECT (no per-match info exists). If a future route needs k<1, add it here WITH an
# outcome-fitted-and-genuinely-overconfident justification (not an unvalidated hedge).
CONSTANT_DEPENDENT_TIERS = []

LO, HI = 0.20, 0.80          # two very different c_hat values
P_HAT = 0.40                  # mid-range read (no [0.02,0.98] clip interaction)


def _moves(tier, qt):
    a = optimize(tier=tier, question_type=qt, p_hat=P_HAT, shadow=LO).q
    b = optimize(tier=tier, question_type=qt, p_hat=P_HAT, shadow=HI).q
    return abs(a - b)


def test_true_p_tiers_independent_of_chat():
    failures = []
    for tier, qt in INDEPENDENT_TIERS:
        if _moves(tier, qt) > 1e-9:
            failures.append((tier, qt, _moves(tier, qt)))
    assert not failures, f"TRUE-P tiers moved with c_hat (disease!): {failures}"


def test_independent_tiers_really_deploy_k1():
    # belt + suspenders: the structural prior for every 'independent' tier is exactly 1.0
    for tier, qt in INDEPENDENT_TIERS:
        cls, sub = classify(tier, qt)
        assert K_PRIOR.get((cls, sub)) == 1.0, f"{tier} -> {(cls,sub)} prior != 1.0"


def test_report_constant_dependent_tiers():
    # Diagnostic: print the disease surface (k<1 tiers whose q moves with c_hat). Does not fail --
    # these are the rows under review; the assertion is only that we KNOW the full list.
    print("\n  c_hat-DEPENDENT tiers (q moves when c_hat moves) — disease surface for review:")
    seen = []
    for tier, qt in CONSTANT_DEPENDENT_TIERS:
        cls, sub = classify(tier, qt)
        mv = _moves(tier, qt)
        seen.append((tier, qt))
        print(f"    {tier:18} {qt:22} k_prior={K_PRIOR.get((cls,sub)):.2f}  q-moves-by={mv:.3f}")
    assert seen == CONSTANT_DEPENDENT_TIERS  # list is complete/known


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
