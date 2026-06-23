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
    assert "pipe_vs_manual" in rep.columns and "rbp_manual" in rep.columns


def _temp_log():
    import tempfile, os
    # a path that does NOT yet exist, so log_slate writes the header
    return os.path.join(tempfile.mkdtemp(), "log.csv")


def test_override_records_and_keeps_pipeline_counterfactual():
    from odds_lib.measurement import log_slate, apply_override
    import pandas as pd
    sheet = pd.DataFrame([{"match": "A vs B", "q": "Q8", "type": "team_sot_over",
                           "tier": "RATE_SOT", "p_hat": 0.669, "shadow": 0.48, "SUBMIT": 0.669}])
    path = _temp_log()
    log_slate(sheet, run_id="r1", path=path)
    n = apply_override("r1", "A vs B", "Q8", 0.60,
                       "near threshold; slope likely overstates", path=path)
    assert n == 1
    row = pd.read_csv(path).iloc[0]
    assert float(row["final_submitted"]) == 0.60   # what we'll score
    assert float(row["pipeline_submit"]) == 0.669  # kept as counterfactual
    assert row["source"] == "manual" and str(row["override_reason"]).strip()


def test_override_requires_reason():
    from odds_lib.measurement import log_slate, apply_override
    import pandas as pd
    sheet = pd.DataFrame([{"match": "A vs B", "q": "Q8", "type": "x", "tier": "RATE_SOT",
                           "p_hat": 0.6, "shadow": 0.5, "SUBMIT": 0.6}])
    path = _temp_log()
    log_slate(sheet, run_id="r1", path=path)
    try:
        apply_override("r1", "A vs B", "Q8", 0.5, "  ", path=path)
        assert False, "should have refused empty reason"
    except ValueError:
        pass


def test_classify_override_buckets():
    from odds_lib.measurement import classify_override
    assert classify_override("") == "unlabeled"
    assert classify_override("player not_starting; removed void") == "hard_qa"
    assert classify_override("pipeline_default_rounded") == "rounding_or_default"
    assert classify_override("felt high so manual cap toward shadow") == "soft"
    assert classify_override("role_sensitive favorite should control territory") == "soft"
    # SOFT wins over an incidental hard-QA token (the ACTION is a soft trim)
    assert classify_override("modest human trim felt high but mahrez starting no_hard_fade") == "soft"
    # a non-soft, non-hard rationale (e.g. the slope argument) is 'other', allowed
    assert classify_override("near threshold; slope likely overstates") == "other"


def test_soft_override_disabled_by_default():
    from odds_lib.measurement import log_slate, apply_override
    import pandas as pd
    sheet = pd.DataFrame([{"match": "A vs B", "q": "Q8", "type": "x", "tier": "RATE_SOT",
                           "p_hat": 0.7, "shadow": 0.5, "SUBMIT": 0.7}])
    path = _temp_log()
    log_slate(sheet, run_id="r1", path=path)
    # SOFT reason refused by default ...
    try:
        apply_override("r1", "A vs B", "Q8", 0.55, "felt high, manual cap toward middle", path=path)
        assert False, "soft override should be disabled by default"
    except ValueError:
        pass
    # ... but allow_soft makes a deliberate, logged exception
    n = apply_override("r1", "A vs B", "Q8", 0.55, "felt high, manual cap toward middle",
                       allow_soft=True, path=path)
    assert n == 1
    # HARD_QA fix is always allowed
    n2 = apply_override("r1", "A vs B", "Q8", 0.02, "player not_starting; void", path=path)
    assert n2 == 1


def test_manual_aggregated_only_where_exists_not_imputed_zero():
    import math
    from odds_lib.measurement import match_report
    # group X: one row with manual, one without -> manual stats over 1 row only
    dfx = pd.DataFrame([
        _row(tier="X", final_submitted="0.7", pipeline_submit="0.7", manual_estimate="0.5",
             actual_rbp="11", result="yes", shadow="0.6"),
        _row(tier="X", final_submitted="0.3", pipeline_submit="0.3",  # no manual
             actual_rbp="-5", result="yes", shadow="0.5"),
    ])
    rep = tier_report(score_rows(dfx))
    assert rep.loc["X", "n_man"] == 1 and pd.notna(rep.loc["X", "rbp_manual"])
    # group with zero manual rows -> NaN, never 0
    dfy = pd.DataFrame([_row(tier="Y", final_submitted="0.7", pipeline_submit="0.7",
                             actual_rbp="11", result="yes", shadow="0.6")])
    rep2 = tier_report(score_rows(dfy))
    assert rep2.loc["Y", "n_man"] == 0 and math.isnan(rep2.loc["Y", "rbp_manual"])
    # match_report runs and totals rbp_final
    mrep = match_report(score_rows(dfx))
    assert "A vs B" in mrep.index and abs(mrep.loc["A vs B", "rbp_final"] - 6.0) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
