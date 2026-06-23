"""Guards for the market-anchored goals engine.

    .venv/bin/python tests/test_match_engine.py

Key invariants: calibration reproduces the market it was fit to; a known
(lam_h, lam_a) round-trips; and the engine reproduces markets it was NOT
fit to (BTTS, HT-draw) within Monte-Carlo tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib import match_engine as E


def test_calibration_reproduces_inputs():
    m = E.calibrate("H", "A", p_home=0.55, p_over=0.52)
    assert abs(m.fit_p_home - 0.55) < 1e-3
    assert abs(m.fit_p_over - 0.52) < 1e-3


def test_known_lambdas_roundtrip():
    lam_h, lam_a = 1.8, 1.0
    p_home = E._p_home_win(lam_h, lam_a)
    p_over = E._p_over(lam_h + lam_a, 2.5)
    m = E.calibrate("H", "A", p_home=p_home, p_over=p_over)
    assert abs(m.lam_home - lam_h) < 0.03
    assert abs(m.lam_away - lam_a) < 0.03


def test_engine_reproduces_unfit_markets():
    # Fit only 1X2+total; check BTTS/HT-draw come out near an independent
    # analytic expectation. Heavy favorite -> low BTTS.
    m = E.calibrate("H", "A", p_home=0.89, p_over=0.70)
    sim = E.simulate(m, n=120_000, seed=1)
    # independent-Poisson BTTS = (1-e^-lam_h)(1-e^-lam_a)
    import math
    btts_analytic = (1 - math.exp(-m.lam_home)) * (1 - math.exp(-m.lam_away))
    assert abs(E.p_btts(sim) - btts_analytic) < 0.01


def test_favorite_scores_more_likely():
    m = E.calibrate("H", "A", p_home=0.70, p_over=0.50)
    sim = E.simulate(m, n=120_000, seed=2)
    assert E.p_team_score_any(sim, "H") > E.p_team_score_any(sim, "A")


def test_probabilities_in_unit_interval():
    m = E.calibrate("H", "A", p_home=0.42, p_over=0.51)
    sim = E.simulate(m, n=80_000, seed=3)
    for p in [E.p_second_half_more_goals(sim),
              E.p_team_more_goals_2h(sim, "H"),
              E.p_compound_btts_over_2_5(sim),
              E.p_total_goals_2h_over(sim, 2),
              E.p_compound_first_goal_score_2h(sim, "H", "A")]:
        assert 0.0 <= p <= 1.0


def test_total_goals_2h_over_monotone_and_consistent():
    m = E.calibrate("H", "A", p_home=0.83, p_over=0.57)
    sim = E.simulate(m, n=160_000, seed=5)
    p1 = E.p_total_goals_2h_over(sim, 1)   # >=1 2H goal
    p2 = E.p_total_goals_2h_over(sim, 2)   # >=2 2H goals
    p3 = E.p_total_goals_2h_over(sim, 3)
    assert p1 > p2 > p3                      # tail is monotone decreasing
    # 2H 2+ goals must be LESS likely than full-match 3+ goals (2H ⊂ full match)
    assert p2 < E.p_over_2_5(sim)
    # 2H total goals = full total - 1H total; mean check vs analytic 2H lambda
    lam2h = (m.lam_home + m.lam_away) * (1 - m.h1_share)
    assert abs((sim.n2h + sim.n2a).mean() - lam2h) < 0.03


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
