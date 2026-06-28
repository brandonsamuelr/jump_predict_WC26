"""Guards for the hydration time-window helper + advance derivation.

    .venv/bin/python tests/test_time_window.py

Enforces: break-fact GATE (no unverified break silently prices), NO caps (a founded high
prob survives), moderation comes ONLY from modeled share uncertainty (posterior mean, not a
ceiling or pull-to-middle), and advance includes ET/penalty paths.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib.time_window import (
    window_probability, share_distribution, deterministic_window_p, BreakNotVerified)
from odds_lib.advance_market import p_advance, win_if_drawn_from_edge, price_advance


def test_break_fact_gate_refuses_unverified():
    try:
        window_probability(2.0, [(0.3, 1.0)], breaks_verified=False)
        assert False, "must raise when the break fact is not verified"
    except BreakNotVerified:
        pass


def test_no_cap_positive_founded_high_prob_survives():
    # high volume in a window with a high share -> high posterior mean; must NOT be capped.
    lam, s = 6.0, 0.5                                  # 1 - e^{-3} ~ 0.950
    p = window_probability(lam, share_distribution(s, 0.0))
    assert p > 0.90                                    # survives, no 0.75/0.80 ceiling
    assert abs(p - deterministic_window_p(lam, s)) < 1e-9
    # a still-higher founded input yields a still-higher output (no ceiling anywhere)
    assert window_probability(9.0, share_distribution(0.6, 0.0)) > p


def test_posterior_mean_moderation_only_from_uncertainty():
    lam, s = 4.0, 0.5                                  # certain ~ 0.865 (far from 0.5)
    certain = window_probability(lam, share_distribution(s, 0.0))
    assert abs(certain - deterministic_window_p(lam, s)) < 1e-9     # certain == deterministic calc
    atoms = share_distribution(s, 0.10)
    uncertain = window_probability(lam, atoms)
    # concavity of 1-e^{-x}: a WIDER share dist lowers the posterior mean -- by the model
    assert uncertain < certain
    # it is EXACTLY the weighted atom average (not a cap, not a generic pull-to-0.5)
    manual = (sum(w * (1 - math.exp(-lam * ss)) for ss, w in atoms)
              / sum(w for _, w in atoms))
    assert abs(uncertain - manual) < 1e-12
    # moderation is small + founded, NOT a collapse toward 0.5
    assert abs(uncertain - certain) < abs(uncertain - 0.5)


def test_advance_includes_et_and_penalty_paths():
    pw, pd = 0.45, 0.27
    pa = p_advance(pw, pd, p_win_if_drawn=0.5)
    assert pa >= pw and pa > pw                         # ET/pens strictly ADD paths (draw>0)
    assert abs(pa - (pw + pd * 0.5)) < 1e-12
    # degenerate: no draw mass -> advance == regulation win
    assert abs(p_advance(0.6, 0.0, 0.5) - 0.6) < 1e-12
    # favorite tilt is bounded and odds-derived (never a typed/crowd constant)
    assert 0.5 < win_if_drawn_from_edge(0.55, 0.20, strength=0.5) < 1.0


def test_advance_prohibits_flat_half_output_is_odds_derived():
    # asymmetric favorite (Brazil-ish): P(win|draw) is ODDS-DERIVED, OUTPUT strictly off 0.5
    fav = price_advance(p_win_regulation=0.62, p_regulation_draw=0.22, p_opp_win_regulation=0.16)
    assert fav["priced"] and fav["conditional_source"] == "odds_derived_tilt"
    assert fav["conditional"] > 0.5 and abs(fav["conditional"] - 0.5) > 1e-6   # the NUMBER, not "tilt called"
    assert fav["p_advance"] > 0.62                                             # ET/pens add paths
    # underdog side strictly below 0.5
    dog = price_advance(p_win_regulation=0.16, p_regulation_draw=0.22, p_opp_win_regulation=0.62)
    assert dog["conditional"] < 0.5 and abs(dog["conditional"] - 0.5) > 1e-6


def test_advance_strength_zero_loophole_closed():
    # strength=0 makes win_if_drawn_from_edge return 0.5 regardless of edge -> must be REJECTED,
    # so flat 0.5 cannot sneak back in through the tilt function on an asymmetric match.
    try:
        price_advance(0.62, 0.22, 0.16, strength=0.0)
        assert False, "strength<=0 must raise (flat-0.5 prohibition)"
    except ValueError:
        pass


def test_advance_symmetric_is_allowed_but_flagged():
    sym = price_advance(0.40, 0.30, 0.40)                  # genuinely symmetric by the odds win-edge
    assert sym["conditional"] == 0.5 and sym["conditional_source"] == "symmetric_odds_derived"


def test_advance_no_strength_signal_flags_row_not_half():
    miss = price_advance(p_win_regulation=None, p_regulation_draw=0.25, p_opp_win_regulation=None)
    assert miss["priced"] is False and miss["conditional"] is None and miss["flag"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
