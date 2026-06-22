"""Guards for the breadth-first submission optimizer + field shadow anchor.

    .venv/bin/python tests/test_optimizer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib.optimizer import optimize
from odds_lib.field_model import FieldMeanEstimator


def test_market_submits_p_hat():
    s = optimize(tier="MARKET", p_hat=0.89, shadow=0.50)
    assert s.mode == "lean" and abs(s.q - 0.89) < 1e-9


def test_no_p_hat_shadows():
    s = optimize(tier="PENDING", p_hat=None, shadow=0.389)
    assert s.mode == "shadow" and abs(s.q - 0.389) < 1e-9


def test_sot_comparison_is_trusted():
    # RATE_SOT_CMP (constant cancels in a comparison) is trusted -> submit p_hat,
    # NOT the role-blind shadow.
    s = optimize(tier="RATE_SOT_CMP", p_hat=0.194, shadow=0.542)
    assert s.mode == "lean" and abs(s.q - 0.194) < 1e-9


def test_untrusted_tier_shadows_not_hedged():
    # A no-model row (PENDING) shadows the field, with no hedge toward 0.5.
    s = optimize(tier="PENDING", p_hat=None, shadow=0.48)
    assert s.mode == "shadow" and abs(s.q - 0.48) < 1e-9


def test_calibrated_sot_count_is_trusted():
    # RATE_SOT is now pooled-calibrated -> trusted, submits the model p_hat
    # (not the role-blind shadow).
    s = optimize(tier="RATE_SOT", p_hat=0.22, shadow=0.48)
    assert s.mode == "lean" and abs(s.q - 0.22) < 1e-9


def test_variance_tilt_overshoots_p_hat():
    base = optimize(tier="MARKET", p_hat=0.80, shadow=0.5).q
    tilted = optimize(tier="MARKET", p_hat=0.80, shadow=0.5, variance_tilt=0.2).q
    assert tilted > base  # further from 0.5, beyond p_hat


def test_clip_bounds():
    s = optimize(tier="MARKET", p_hat=0.999, shadow=0.5, variance_tilt=0.5)
    assert s.q <= 0.97


def test_unknown_tier_treated_as_shadow():
    s = optimize(tier="WHATEVER", p_hat=0.7, shadow=0.4)
    assert s.mode == "shadow" and abs(s.q - 0.4) < 1e-9


def test_field_estimator_known_vs_unknown():
    fe = FieldMeanEstimator()
    known = fe.estimate("penalty_or_red_card")
    assert known.source == "qt_mean" and 0.3 < known.q_hat < 0.5
    unknown = fe.estimate("totally_made_up_type")
    assert unknown.source == "global_mean"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
