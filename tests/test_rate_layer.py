"""Guards for the total_sot_2h_over level recenter (2026-06-23).

The surgical fix must: recenter the total row's level DOWN, PRESERVE per-match
tilt (not re-freeze to a constant), and leave single-team / comparison SOT
(which share the constants) untouched.

    .venv/bin/python tests/test_rate_layer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib import rate_layer as R

H1 = 0.45


def test_total_ships_raw_no_offset():
    # FIXED 2026-06-26: the -1.2 field-pull offset is removed. price_total_sot_2h_over now ships
    # the RAW market-anchored Poisson tail (no recenter toward the ~0.62 field placeholder).
    rr = R.price_total_sot_2h_over(1.48, 1.24, 3.5, H1)
    raw_mu = R.team_sot_mu(1.48, 1 - H1) + R.team_sot_mu(1.24, 1 - H1)
    assert abs(rr.p - R._p_ge(raw_mu, 4)) < 1e-9   # equals the raw output (offset gone)
    assert R.TOTAL_SOT_2H_LEVEL_OFFSET == 0.0      # constant deprecated to 0


def test_total_still_tilts_not_frozen():
    hi = R.price_total_sot_2h_over(1.8, 1.8, 3.5, H1).p   # high tempo
    lo = R.price_total_sot_2h_over(0.8, 0.8, 3.5, H1).p   # cagey
    assert hi - lo > 0.3                          # real matchup variation remains


def test_single_team_sot_concave_map():
    # CONCAVE map (2026-06-26): values recomputed under team_sot_mu = A*(1-exp(-B*lam)).
    assert abs(R.price_team_sot_over(1.90, 5.5).p - 0.490) < 1e-3
    assert abs(R.price_team_sot_over(0.45, 3.5).p - 0.083) < 1e-3
    cmp = R.price_team_more_sot_2h(0.77, 1.90, H1).p
    assert 0.05 < cmp < 0.30   # comparison still favours the higher-lambda team


def test_concave_map_saturates_and_tilts():
    # the saturating map keeps a real matchup tilt but no high-lambda over-extrapolation
    assert R.team_sot_mu(2.95) < 8.0                         # bounded (linear gave 9.9)
    hi = R.price_total_sot_2h_over(1.8, 1.8, 3.5, H1).p
    lo = R.price_total_sot_2h_over(0.8, 0.8, 3.5, H1).p
    assert hi - lo > 0.3                                     # matchup variation preserved


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
