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


def test_total_recentered_below_raw():
    rr = R.price_total_sot_2h_over(1.48, 1.24, 3.5, H1)  # Norway-Senegal
    raw_mu = R.team_sot_mu(1.48, 1 - H1) + R.team_sot_mu(1.24, 1 - H1)
    assert rr.p < R._p_ge(raw_mu, 4)              # lower than the old raw output
    assert 0.55 < rr.p < 0.72                     # lands ~at the field level


def test_total_still_tilts_not_frozen():
    hi = R.price_total_sot_2h_over(1.8, 1.8, 3.5, H1).p   # high tempo
    lo = R.price_total_sot_2h_over(0.8, 0.8, 3.5, H1).p   # cagey
    assert hi - lo > 0.3                          # real matchup variation remains


def test_single_team_sot_untouched():
    # The offset must NOT leak into single-team or comparison rows.
    assert abs(R.price_team_sot_over(1.90, 5.5).p - 0.669) < 1e-3
    assert abs(R.price_team_sot_over(0.45, 3.5).p - 0.216) < 1e-3
    # comparison uses the same constants and must be unchanged too
    cmp = R.price_team_more_sot_2h(0.77, 1.90, H1).p
    assert 0.10 < cmp < 0.30


def test_offset_does_not_touch_shared_constants():
    assert R.SOT_INTERCEPT == 1.01 and R.SOT_SLOPE == 3.03   # unchanged


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
