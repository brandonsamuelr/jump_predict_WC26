"""Per-tier measurement loop — settle "is it better than the field / my LLM /
the pipeline" with data instead of vibes.

The contest scores actual_rbp against whatever you ACTUALLY locked, so:

    field_Brier = final_Brier + actual_rbp / (100 * m)        [recovered]
    RBP(q') = 100 * m * (field_Brier - (q' - y)^2)            [any candidate]

where final_Brier uses ``final_submitted`` (your real locked number, which
may differ from the pipeline's value after a manual override). We then score
every candidate on the same recovered field:

    final      - what you actually submitted (== actual_rbp)
    pipeline   - what the optimizer produced (counterfactual if you'd trusted it)
    p_hat      - the raw model probability
    shadow     - the field-mean anchor (always-shadow strategy)
    llm        - your hand/LLM estimate

Aggregated by tier. No new predictor here — this only measures the ones we have.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LOG_PATH = Path("data/measurement_log.csv")

COLS = [
    "run_id", "match", "question_number", "question_type", "tier", "source",
    "p_hat", "shadow", "llm_estimate", "pipeline_submit", "final_submitted",
    "multiplier",
    # filled post-resolution:
    "result", "actual_rbp", "field_prob",
]


def log_slate(sheet: pd.DataFrame, run_id: str, multiplier: int = 1,
              path: Path | str = LOG_PATH) -> int:
    """Append an optimizer submit sheet to the measurement log.

    ``final_submitted`` defaults to the pipeline value; edit it in the CSV if
    you override a row at lock (and set ``source`` to manual/llm). Post-
    resolution columns are left blank.
    """
    out = pd.DataFrame({
        "run_id": run_id,
        "match": sheet["match"],
        "question_number": sheet["q"],
        "question_type": sheet["type"],
        "tier": sheet["tier"],
        "source": "pipeline",
        "p_hat": sheet.get("p_hat", ""),
        "shadow": sheet.get("shadow", ""),
        "llm_estimate": sheet.get("llm_estimate", ""),
        "pipeline_submit": sheet["SUBMIT"],
        "final_submitted": sheet["SUBMIT"],
        "multiplier": multiplier,
        "result": "", "actual_rbp": "", "field_prob": "",
    })
    path = Path(path)
    header = not path.exists()
    out.reindex(columns=COLS).to_csv(path, mode="a", header=header, index=False)
    return len(out)


def _to_y(result) -> float | None:
    s = str(result).strip().lower()
    if s in ("1", "yes", "y", "true"):
        return 1.0
    if s in ("0", "no", "n", "false"):
        return 0.0
    return None


def _num(v):
    try:
        x = float(v)
        return x if not np.isnan(x) else None
    except (TypeError, ValueError):
        return None


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Score resolved rows: recover field Brier from the FINAL submission and
    actual_rbp, then RBP every candidate. A row is scored only if it has a
    result, actual_rbp, and a final submission."""
    rows = []
    for _, r in df.iterrows():
        y = _to_y(r.get("result"))
        arbp = _num(r.get("actual_rbp"))
        final = _num(r.get("final_submitted"))
        if final is None:
            final = _num(r.get("pipeline_submit"))
        if y is None or arbp is None or final is None:
            continue
        m = _num(r.get("multiplier")) or 1.0
        field_brier = (final - y) ** 2 + arbp / (100.0 * m)

        def rbp(qp):
            qp = _num(qp)
            return None if qp is None else round(100.0 * m * (field_brier - (qp - y) ** 2), 3)

        rows.append({
            "match": r.get("match"), "question_number": r.get("question_number"),
            "question_type": r.get("question_type"), "tier": r.get("tier"),
            "source": r.get("source"), "y": y, "multiplier": m,
            "rbp_final": round(arbp, 3),
            "rbp_pipeline": rbp(r.get("pipeline_submit")),
            "rbp_phat": rbp(r.get("p_hat")),
            "rbp_shadow": rbp(r.get("shadow")),
            "rbp_llm": rbp(r.get("llm_estimate")),
        })
    return pd.DataFrame(rows)


def tier_report(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by tier: mean RBP per strategy + the headline edges."""
    if scored.empty:
        return pd.DataFrame()

    def _agg(g):
        return pd.Series({
            "n": len(g),
            "rbp_final": round(g["rbp_final"].mean(), 2),
            "beat_field": round((g["rbp_final"] > 0).mean(), 2),
            "rbp_pipeline": round(g["rbp_pipeline"].mean(), 2),
            "rbp_shadow": round(g["rbp_shadow"].mean(), 2),
            "rbp_llm": round(g["rbp_llm"].mean(), 2),
            "pipeline_vs_shadow": round(g["rbp_pipeline"].mean() - g["rbp_shadow"].mean(), 2),
            "pipeline_vs_llm": round(g["rbp_pipeline"].mean() - g["rbp_llm"].mean(), 2),
        })

    rep = scored.groupby("tier").apply(_agg, include_groups=False)
    rep.loc["ALL"] = _agg(scored)
    return rep


__all__ = ["LOG_PATH", "COLS", "log_slate", "score_rows", "tier_report"]
