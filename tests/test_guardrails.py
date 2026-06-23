"""Guards for the visibility/guardrail reports (Tasks 1-3).

    .venv/bin/python tests/test_guardrails.py

Reports FLAG; they never correct. Soft overrides stay disabled by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from crowd_reliability_report import crowd_error, prep, reliability_table
from low_information_exposure_report import flag_row


# --- Task 1: reliability audit -------------------------------------------

def test_crowd_error_math_percent_and_fraction():
    assert abs(crowd_error(0.478, 0.35) - 0.128) < 1e-9   # fraction input
    assert abs(crowd_error(0.478, 35) - 0.128) < 1e-9     # percent input normalizes


def test_reliability_table_groups_with_counts():
    df = pd.DataFrame([
        {"match": "A vs B", "question_number": "Q1", "question_type": "player_sot_2h_over",
         "tier": "PENDING", "shadow": "0.478", "crowd_prob": "35", "final_submitted": "0.478",
         "result": "", "actual_rbp": ""},
        {"match": "C vs D", "question_number": "Q1", "question_type": "player_sot_2h_over",
         "tier": "PENDING", "shadow": "0.50", "crowd_prob": "40", "final_submitted": "0.50",
         "result": "", "actual_rbp": ""},
    ])
    t = reliability_table(prep(df), "question_type")
    assert "player_sot_2h_over" in t.index and "ALL" in t.index
    for col in ("n_rows", "n_match_clusters", "mean_abs_error", "max_abs_error"):
        assert col in t.columns
    assert t.loc["player_sot_2h_over", "n_rows"] == 2
    assert t.loc["player_sot_2h_over", "n_match_clusters"] == 2


# --- Task 2: low-information exposure (flags only) ------------------------

def test_flag_pending_player_sot_2h_high():
    flags, stake = flag_row(question_type="player_sot_2h_over", tier="PENDING", cls="SHADOW",
                            mode="shadow", c_hat=0.478, p_hat=None, submitted_prob=0.478,
                            underdog=False, expected_err=0.088)
    assert "NO_MODEL" in flags and "PENDING_OR_SHADOW" in flags
    assert "SUBMIT_ABOVE_SANITY_CAP" in flags        # 0.478 > 0.40
    assert "HIGH_STAKE_LOW_INFO" in flags            # 200*0.088 = 17.6 >= 15
    assert stake is not None and stake >= 15


def test_flag_underdog_2h_sot_above_cap():
    flags, _ = flag_row(question_type="player_sot_2h_over", tier="PENDING", cls="SHADOW",
                        mode="shadow", c_hat=0.34, p_hat=None, submitted_prob=0.34,
                        underdog=True, expected_err=0.088)
    assert "UNDERDOG_PLAYER_2H_SOT_HIGH" in flags     # 0.34 > 0.325 underdog cap
    assert "SUBMIT_ABOVE_SANITY_CAP" in flags


def test_flag_benched_player_goal():
    flags, _ = flag_row(question_type="player_goal", tier="PENDING", cls="SHADOW",
                        mode="shadow", c_hat=0.36, p_hat=None, submitted_prob=0.15,
                        lineup_status="bench_unknown", expected_err=0.065)
    assert "BENCH_PLAYER" in flags
    assert "SUBMIT_ABOVE_SANITY_CAP" in flags         # benched player_goal 0.15 > 0.12


def test_does_not_flag_trusted_market_engine_deviation():
    # ANTI-SPAM: a MARKET row that deviates from c_hat (has p_hat) is NOT flagged.
    flags, _ = flag_row(question_type="team_win", tier="MARKET", cls="MARKET", mode="edge",
                        c_hat=0.621, p_hat=0.838, submitted_prob=0.817, expected_err=0.16)
    assert flags == []
    flags2, _ = flag_row(question_type="compound_btts_over_2_5", tier="ENGINE_GOALS",
                         cls="ENGINE", mode="edge", c_hat=0.392, p_hat=0.354,
                         submitted_prob=0.358, expected_err=0.057)
    assert flags2 == []


def test_sot_anchor_pull_flag():
    flags, _ = flag_row(question_type="team_sot_over", tier="RATE_SOT", cls="RATE_SOT",
                        mode="edge", c_hat=0.481, p_hat=0.292, submitted_prob=0.386,
                        expected_err=0.133)
    assert "SOT_ANCHOR_PULL" in flags                 # moved +0.094 from p_hat toward c_hat

def test_sot_anchor_pull_not_fired_when_small():
    flags, _ = flag_row(question_type="team_sot_over", tier="RATE_SOT", cls="RATE_SOT",
                        mode="edge", c_hat=0.50, p_hat=0.47, submitted_prob=0.485,
                        expected_err=0.133)
    assert "SOT_ANCHOR_PULL" not in flags             # |0.485-0.47|=0.015 < 0.07


def test_confirmed_starter_thin_market_not_unconfirmed():
    # Q9-type cosmetic fix: confirmed starter on a thin market (PROP_thin) is a
    # THIN_MARKET note, not an UNCONFIRMED-lineup cry-wolf.
    flags, _ = flag_row(question_type="player_sot_over", tier="PROP_thin", cls="PROP",
                        mode="edge", c_hat=0.471, p_hat=0.43, submitted_prob=0.455,
                        lineup_status="starter", expected_err=0.021)
    assert "PLAYER_PROP_THIN_MARKET" in flags
    assert "PLAYER_PROP_UNCONFIRMED" not in flags


def test_is_lower_bound_prop_only_goal_or_assist():
    from odds_lib.player_prop_pricing import is_lower_bound_prop
    assert is_lower_bound_prop("player_goal_or_assist") is True
    assert is_lower_bound_prop("player_goal") is False
    assert is_lower_bound_prop("player_sot_over") is False
    assert is_lower_bound_prop("player_sot_2h_over") is False   # None spec


def test_lower_bound_clamp_reason_is_hard_qa_not_soft():
    from odds_lib.measurement import classify_override
    assert classify_override("LOWER_BOUND_CLAMP -> p_hat; measurement-invariant") == "hard_qa"
    assert classify_override("goal_or_assist lower_bound floor raise; measurement-bias") == "hard_qa"


# --- Task 3: soft overrides remain disabled ------------------------------

def test_soft_overrides_disabled_by_default():
    from odds_lib.measurement import log_slate, apply_override
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "log.csv")
    sheet = pd.DataFrame([{"match": "A vs B", "q": "Q1", "type": "x", "tier": "RATE_SOT",
                           "p_hat": 0.7, "shadow": 0.5, "SUBMIT": 0.7}])
    log_slate(sheet, run_id="r1", path=path)
    try:
        apply_override("r1", "A vs B", "Q1", 0.55, "felt high, manual cap toward middle", path=path)
        assert False, "soft override should be disabled by default"
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
