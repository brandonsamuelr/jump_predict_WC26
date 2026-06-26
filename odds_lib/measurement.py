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
    "run_id", "match", "question_number", "question_type", "tier",
    "weak_row_class", "source",
    "p_hat", "shadow", "manual_estimate", "llm_estimate",
    "pipeline_submit", "final_submitted", "override_reason", "multiplier",
    # filled post-resolution (crowd_prob/if_yes_rbp/if_no_rbp/at_stake from the
    # locked sheet; result/actual_rbp after the match):
    "result", "actual_rbp", "crowd_prob", "if_yes_rbp", "if_no_rbp", "at_stake",
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
        "weak_row_class": sheet.get("weak_row_class", ""),
        "source": "pipeline",
        "p_hat": sheet.get("p_hat", ""),
        "shadow": sheet.get("shadow", ""),
        "manual_estimate": sheet.get("manual_estimate", ""),
        "llm_estimate": sheet.get("llm_estimate", ""),
        "pipeline_submit": sheet["SUBMIT"],
        "final_submitted": sheet["SUBMIT"],
        "override_reason": "",
        "multiplier": multiplier,
        "result": "", "actual_rbp": "", "crowd_prob": "",
        "if_yes_rbp": "", "if_no_rbp": "", "at_stake": "",
    })
    path = Path(path)
    header = not path.exists()
    out.reindex(columns=COLS).to_csv(path, mode="a", header=header, index=False)
    return len(out)


# --- Override leakage policy (Deliverable 2) --------------------------------
# Manual overrides have historically REDUCED score (~+4.17/q raw vs ~+3.24/q
# after overrides => ~0.93/q leak — about our needed catch-up rate). The leak is
# the SOFT category: human distrust of a confident model, pulling a trusted row
# toward the middle, or vague game-script intuition. Those are DISABLED by
# default. HARD_QA (factual corrections) are kept. Reasons are free text, so the
# classifier is a documented heuristic; the D2 report prints every row's
# (reason, category, delta, leakage) so any misclassification is auditable.
HARD_QA_PATTERNS = (
    "not_starting", "not starting", "void", "removed", "coverage_lost",
    "wrong_player", "wrong player", "wrong_threshold", "wrong threshold",
    "wrong_line", "stale", "mapping", "no_cut", "no_hard_fade",
    # confirmed bench is a lineup FACT (same class as not_starting): a benched
    # player's prop is a wrong input, not soft distrust.
    "bench", "confirmed_bench",
    # definitional / measurement invariants (e.g. the lower-bound clamp) — these
    # are arithmetic facts, NOT soft distrust, so they must categorize hard_qa
    # and never count as soft-override leakage.
    "lower_bound", "lower bound", "clamp", "measurement_invariant",
    "measurement-invariant", "measurement_bias", "measurement-bias",
)
# Recording errors (NOT decisions) — must be excluded from strategy analysis.
ENTRY_SHIFT_PATTERNS = ("entry_shift", "entry_error")
# DELIVERY-PATH failures (NOT decisions, NOT estimates) — the submitted value was
# forced by a process failure (refresh missed/crashed the lock -> coin-flips, or a
# slate never reached the board). Like entry_shift, these are excluded from strategy
# analysis: they must NOT count as pipeline/model performance in the pace/tier stats.
DELIVERY_FAIL_PATTERNS = ("refresh_missed", "missed_lock", "coinflip", "coin_flip",
                          "refresh_crash", "attempted_not_submitted", "not_submitted",
                          "delivery_fail")
# Empirical (measured-bias) trims, e.g. SOT high-bias. Distinct from soft distrust
# AND from hard_qa: an empirical bet to be MEASURED separately (may pay or not).
EMPIRICAL_TRIM_PATTERNS = ("empirical_sot_trim", "sot_trim", "empirical_trim")
SOFT_PATTERNS = (
    "felt", "feels", "intuition", "overtrust", "split_difference",
    "split the difference", "manual_cap", "_cap", " cap", "trim", "toward_manual",
    "moderate_shadow_toward", "role_sensitive", "favorite_should", "game_script",
    "physical", "chasing", "control_territory", "possession", "likely_high",
    "keep_close", "keep close",
)


def classify_override(reason: str) -> str:
    """Bucket an override reason into one of: 'entry_shift' (a RECORDING error,
    not a decision — excluded from strategy analysis), 'delivery_failure' (a
    delivery-path failure — coin-flip/missed/crashed lock; also excluded from
    strategy analysis, must not count as pipeline performance), 'empirical_trim' (a
    measured-bias trim, tracked separately to see if it pays), 'soft'
    (net-negative distrust, disabled by default), 'hard_qa' (factual/definitional
    correction, kept), 'rounding_or_default', 'unlabeled', or 'other'.

    Order matters: entry_shift and empirical_trim are checked BEFORE soft because
    an empirical SOT trim literally contains 'trim' (a soft token) but is a
    distinct, tracked category; and a recording error is not a decision at all.
    SOFT is checked before HARD_QA so a 'modest trim ... felt high' soft override
    isn't rescued by an incidental hard-QA token."""
    r = (reason or "").strip().lower()
    if not r:
        return "unlabeled"
    if any(p in r for p in ENTRY_SHIFT_PATTERNS):
        return "entry_shift"
    # delivery-path failures BEFORE the round/default check: the coin-flip tag
    # contains "default" but it is a delivery failure, not a rounding default.
    if any(p in r for p in DELIVERY_FAIL_PATTERNS):
        return "delivery_failure"
    if any(p in r for p in EMPIRICAL_TRIM_PATTERNS):
        return "empirical_trim"
    if any(p in r for p in SOFT_PATTERNS):
        return "soft"
    if any(p in r for p in HARD_QA_PATTERNS):
        return "hard_qa"
    if "round" in r or "default" in r:
        return "rounding_or_default"
    return "other"


def apply_override(run_id: str, match: str, question_number: str, final: float,
                   reason: str, source: str = "manual", allow_soft: bool = False,
                   path: Path | str = LOG_PATH) -> int:
    """Record a manual deviation: set final_submitted + source + override_reason
    for the matching logged row(s). pipeline_submit is preserved as the
    counterfactual. Refuses an empty reason — the discipline gate. Also refuses
    a SOFT-category reason unless allow_soft=True (the net-negative category is
    disabled by default; pass allow_soft to make a deliberate, logged exception).

    Returns the number of rows updated (0 if not found).
    """
    if not str(reason).strip():
        raise ValueError("override requires a one-sentence reason (discipline gate)")
    if not allow_soft and classify_override(reason) == "soft":
        raise ValueError(
            "SOFT override disabled by default (this category has historically "
            "cost ~0.93/q: human distrust of a confident model / pulling a "
            "trusted row toward the middle / vague game-script intuition). Trust "
            "the model via its edge multiplier k, or pass allow_soft=True for a "
            "deliberate, logged exception. HARD-QA fixes (player not starting, "
            "void, wrong player/threshold/line, stale odds, mapping bug) are kept.")
    p = Path(path)
    df = pd.read_csv(p, dtype=str)
    for c in ("final_submitted", "source", "override_reason"):
        df[c] = df[c].astype(object)
    mask = ((df["run_id"] == str(run_id)) & (df["match"] == match)
            & (df["question_number"] == question_number))
    n = int(mask.sum())
    if n:
        df.loc[mask, "final_submitted"] = str(final)
        df.loc[mask, "source"] = source
        df.loc[mask, "override_reason"] = reason
        df.to_csv(p, index=False)
    return n


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
            return float("nan") if qp is None else round(100.0 * m * (field_brier - (qp - y) ** 2), 3)

        rows.append({
            "match": r.get("match"), "question_number": r.get("question_number"),
            "question_type": r.get("question_type"), "tier": r.get("tier"),
            "source": r.get("source"), "y": y, "multiplier": m,
            "rbp_final": round(arbp, 3),
            "rbp_pipeline": rbp(r.get("pipeline_submit")),
            "rbp_phat": rbp(r.get("p_hat")),
            "rbp_manual": rbp(r.get("manual_estimate")),
            "rbp_shadow": rbp(r.get("shadow")),
            "rbp_llm": rbp(r.get("llm_estimate")),
        })
    return pd.DataFrame(rows)


def build_edge_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Join resolved outcomes to the raw log into the per-row frame the edge
    estimator (:func:`odds_lib.edge.compute_edge_table`) consumes.

    CRITICAL: p_model is the RAW model (the ``p_hat`` column), c_hat is the
    pre-lock field proxy (the ``shadow`` column) — NOT ``pipeline_submit``,
    which after Deliverable 3 is the edge-WEIGHTED output. Fitting k on the
    weighted output would be circular (it already contains k), collapsing the
    estimate toward 1. Here we always fit on the unweighted model.
    """
    from odds_lib.edge import classify
    scored = score_rows(df_raw)
    if scored.empty:
        return pd.DataFrame()
    raw = df_raw.set_index(["match", "question_number"])
    rows = []
    for _, r in scored.iterrows():
        key = (r["match"], r["question_number"])
        if key not in raw.index:
            continue
        src = raw.loc[key]
        if isinstance(src, pd.DataFrame):
            src = src.iloc[0]
        p_model = _num(src.get("p_hat"))
        c_hat = _num(src.get("shadow"))
        if p_model is None or c_hat is None:
            continue
        cls, sub = classify(r["tier"], r["question_type"])
        rows.append({
            "source_class": cls, "source_subtype": sub, "match": r["match"],
            "p_model": p_model, "c_hat": c_hat, "y": r["y"],
            "rbp_final": r["rbp_final"], "rbp_model_cf": r["rbp_pipeline"],
            "rbp_baseline_cf": r["rbp_shadow"],
        })
    return pd.DataFrame(rows)


def _agg_group(g: pd.DataFrame, totals: bool = False) -> pd.Series:
    """Aggregate one group. ``totals`` -> sums (match-level), else means
    (tier-level). manual/LLM are aggregated ONLY over rows where that estimate
    exists (missing is never scored as 0), and the pipeline-vs-X edge is a
    PAIRED diff on that same subset, so comparisons stay apples-to-apples."""
    red = (lambda s: round(s.sum(), 1)) if totals else (lambda s: round(s.mean(), 2))
    gm = g.dropna(subset=["rbp_manual"])
    gl = g.dropna(subset=["rbp_llm"])
    return pd.Series({
        "n": len(g),
        "rbp_final": red(g["rbp_final"]),
        "beat_field": round((g["rbp_final"] > 0).mean(), 2),
        "rbp_pipeline": red(g["rbp_pipeline"]),
        "rbp_shadow": red(g["rbp_shadow"]),
        "pipe_vs_shadow": red(g["rbp_pipeline"] - g["rbp_shadow"]),
        "pipe_vs_final": red(g["rbp_pipeline"] - g["rbp_final"]),
        "n_man": len(gm),
        "rbp_manual": red(gm["rbp_manual"]) if len(gm) else float("nan"),
        "pipe_vs_manual": red(gm["rbp_pipeline"] - gm["rbp_manual"]) if len(gm) else float("nan"),
        "n_llm": len(gl),
        "rbp_llm": red(gl["rbp_llm"]) if len(gl) else float("nan"),
        "pipe_vs_llm": red(gl["rbp_pipeline"] - gl["rbp_llm"]) if len(gl) else float("nan"),
    })


def tier_report(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-tier means (detailed diagnostics). Rows are clustered by match —
    treat these as roughly match-count evidence, not row-count."""
    if scored.empty:
        return pd.DataFrame()
    rep = scored.groupby("tier").apply(lambda g: _agg_group(g, totals=False),
                                       include_groups=False)
    rep.loc["ALL"] = _agg_group(scored, totals=False)
    return rep


def match_report(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-match totals — the honest unit of evidence (questions within a
    match share one game script, so they are correlated). At low match-count
    these per-match rows double as the leave-one-match-out view."""
    if scored.empty:
        return pd.DataFrame()
    rep = scored.groupby("match").apply(lambda g: _agg_group(g, totals=True),
                                        include_groups=False)
    rep.loc["ALL"] = _agg_group(scored, totals=True)
    return rep


__all__ = ["LOG_PATH", "COLS", "log_slate", "apply_override", "classify_override",
           "score_rows", "build_edge_frame", "tier_report", "match_report"]
