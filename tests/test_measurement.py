"""Guards for the measurement loop — field-Brier recovery must use the FINAL
submitted number, and counterfactuals must be self-consistent.

    .venv/bin/python tests/test_measurement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from odds_lib.measurement import score_rows, tier_report


def _row(**kw):
    base = dict(match="A vs B", question_number="Q1", question_type="team_win",
                tier="MARKET", source="pipeline", p_hat="", shadow="",
                llm_estimate="", pipeline_submit="", final_submitted="",
                multiplier="1", result="", actual_rbp="", field_prob="")
    base.update(kw)
    return base


def test_recovers_field_brier_and_counterfactuals():
    # field q-bar=0.6, Var(q)=0.04, y=1, m=1 -> field_Brier=0.20.
    # Submit final=0.7 -> final_Brier=0.09 -> actual_rbp=100*(0.20-0.09)=11.
    df = pd.DataFrame([_row(final_submitted="0.7", pipeline_submit="0.7",
                            actual_rbp="11", result="yes", shadow="0.6")])
    s = score_rows(df).iloc[0]
    assert abs(s["rbp_final"] - 11.0) < 1e-6
    # shadow q=0.6: 100*(0.20-(0.6-1)^2)=100*(0.20-0.16)=4.0
    assert abs(s["rbp_shadow"] - 4.0) < 1e-6


def test_field_brier_uses_final_not_pipeline_after_override():
    # Pipeline said 0.669 but we manually locked 0.60. actual_rbp is scored vs
    # 0.60. Recovery MUST use 0.60; pipeline 0.669 is a counterfactual.
    # y=1,m=1: pick field_Brier=0.30 -> final(0.60)_Brier=0.16 -> rbp=100*(0.30-0.16)=14.
    df = pd.DataFrame([_row(final_submitted="0.60", pipeline_submit="0.669",
                            actual_rbp="14", result="yes", source="manual")])
    s = score_rows(df).iloc[0]
    assert abs(s["rbp_final"] - 14.0) < 1e-6
    # pipeline 0.669: 100*(0.30-(0.669-1)^2)=100*(0.30-0.109561)=19.044
    assert abs(s["rbp_pipeline"] - 19.044) < 1e-2  # here the override HURT us


def test_multiplier_scales():
    df = pd.DataFrame([_row(final_submitted="0.7", pipeline_submit="0.7",
                            actual_rbp="22", result="yes", multiplier="2", shadow="0.6")])
    s = score_rows(df).iloc[0]
    assert abs(s["rbp_shadow"] - 8.0) < 1e-6  # 200*(0.20-0.16)


def test_unresolved_rows_skipped():
    assert score_rows(pd.DataFrame([_row(final_submitted="0.7")])).empty


def test_tier_report_aggregates():
    df = pd.DataFrame([
        _row(tier="MARKET", final_submitted="0.7", pipeline_submit="0.7",
             actual_rbp="11", result="yes", shadow="0.6", llm_estimate="0.65"),
        _row(tier="MARKET", final_submitted="0.3", pipeline_submit="0.3",
             actual_rbp="-5", result="yes", shadow="0.5", llm_estimate="0.4"),
    ])
    rep = tier_report(score_rows(df))
    assert "MARKET" in rep.index and "ALL" in rep.index
    assert rep.loc["MARKET", "n"] == 2
    assert "pipeline_vs_llm" in rep.columns


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
