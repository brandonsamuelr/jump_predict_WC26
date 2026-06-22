"""Player Prop v0 truth scorer + sizing report (standalone research tool).

Reads a filled player-prop collection CSV and, for ``player_sot_over`` and
``player_goal_or_assist`` rows, produces a transparent, data-derived
``p_truth_v0`` (simple Poisson event model) plus a risk-managed
recommended submission. This is decision support for tomorrow's slate, NOT a
production model and NOT wired into the engine.

Discipline
----------
- ``p_truth_v0`` comes from rates × expected minutes, never from a hard-coded
  role/status probability table.
- The ONLY factual override is ``lineup_status == out_of_squad`` ->
  ``p_truth_v0 = 0`` (source ``availability_fact``), because that is an
  availability fact, not a probability.
- Rows missing required inputs route to review with no ``p_truth_v0``.
- Confidence affects SIZING ONLY (shrink + deviation cap), never p_truth.
- Deviation caps are risk policy (prevent overfades), clearly labelled.

Usage::

    python scripts/score_player_prop_edges.py \
        --input data/models/player_prop_collection_template.csv \
        --out data/models/player_prop_edge_report.csv

    python scripts/score_player_prop_edges.py --selftest
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


SUPPORTED_QTS = {"player_sot_over", "player_goal_or_assist"}

# Confidence -> sizing policy. Shrink moves submission from field toward
# p_truth_v0; cap bounds absolute deviation from field. Risk policy, not truth.
SHRINK_BY_CONFIDENCE = {"high": 0.75, "medium": 0.50, "low": 0.25}
MAX_DEV_BY_CONFIDENCE = {"high": 0.18, "medium": 0.12, "low": 0.06}

# Thresholds.
HIGH_SAMPLE = 10
MEDIUM_SAMPLE = 5
LARGE_EDGE = 0.15  # |edge| at/above which we flag large_edge_vs_field

BENCH_STATUSES = {"bench_high_usage", "bench_low_usage", "bench_unknown"}

OUTPUT_COLUMNS = [
    "collection_id",
    "match",
    "target_player",
    "target_team",
    "question_type",
    "line",
    "lineup_status",
    "lineup_role",
    "expected_minutes",
    "rates_sample_matches",
    "rates_source",
    "sot_per90",
    "goals_per90",
    "assists_per90",
    "xg_per90",
    "xa_per90",
    "p_truth_v0",
    "p_truth_xgxa",
    "p_truth_source",
    "truth_confidence",
    "p_field_est",
    "edge_vs_field",
    "shrink_used",
    "max_deviation_cap",
    "recommended_submission",
    "worst_case_rbp_estimate",
    "best_case_rbp_estimate",
    "decision",
    "review_reason",
    "risk_tags",
    "notes",
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _f(row: dict, col: str) -> float | None:
    v = row.get(col)
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "nan", "None", "NaN"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _s(row: dict, col: str) -> str | None:
    v = row.get(col)
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "nan", "None", "NaN"):
        return None
    return s


# ---------------------------------------------------------------------------
# Poisson v0 event model
# ---------------------------------------------------------------------------

def poisson_p_ge_k(lam: float, k: int) -> float:
    """P(X >= k) for X ~ Poisson(lam). k<=0 -> 1.0."""
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    cdf = math.exp(-lam)  # i = 0
    term = cdf
    for i in range(1, k):
        term *= lam / i
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


def _line_to_k(line: float | None) -> int:
    """Threshold count for ``over line``: need X >= floor(line)+1.

    line None or 0.5 -> k=1 (i.e. 1+). line 1.5 -> k=2. line 2.5 -> k=3.
    """
    if line is None:
        return 1
    return int(math.floor(line)) + 1


# ---------------------------------------------------------------------------
# Confidence (sizing only)
# ---------------------------------------------------------------------------

def confidence_tier(
    *,
    status: str | None,
    expected_minutes: float | None,
    minutes_source: str | None,
    sample: float | None,
    rates_source: str | None,
    has_required_rates: bool,
) -> str:
    """high / medium / low / review. Drives sizing, never p_truth."""
    if expected_minutes is None or not has_required_rates:
        return "review"
    n = sample or 0
    # high: starter (or out_of_squad fact), strong sample + sourced minutes/rates
    if (
        status == "starter"
        and n >= HIGH_SAMPLE
        and rates_source
        and minutes_source
    ):
        return "high"
    # medium: a bench status with a usable sample + sourced rates
    if status in BENCH_STATUSES and n >= MEDIUM_SAMPLE and rates_source:
        return "medium"
    # low: minutes present but sample thin or source weak
    return "low"


# ---------------------------------------------------------------------------
# RBP estimate (relative Brier vs field, ~x100 scale matching contest history)
# ---------------------------------------------------------------------------

def _rbp(p_submit: float, p_field: float, outcome: int) -> float:
    field_brier = (p_field - outcome) ** 2
    sub_brier = (p_submit - outcome) ** 2
    return 100.0 * (field_brier - sub_brier)


def rbp_worst_best(p_submit: float, p_field: float) -> tuple[float, float]:
    yes = _rbp(p_submit, p_field, 1)
    no = _rbp(p_submit, p_field, 0)
    return (round(min(yes, no), 2), round(max(yes, no), 2))


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_row(row: dict) -> dict:
    out = {c: "" for c in OUTPUT_COLUMNS}
    # passthrough identity / inputs
    for c in (
        "collection_id", "match", "target_player", "target_team",
        "question_type", "line", "lineup_status", "lineup_role",
        "expected_minutes", "rates_sample_matches", "rates_source",
        "sot_per90", "goals_per90", "assists_per90", "xg_per90", "xa_per90",
        "notes",
    ):
        v = row.get(c)
        out[c] = "" if v is None or str(v).strip().lower() in ("nan", "none") else v

    qt = (_s(row, "question_type") or "").lower()
    status = (_s(row, "lineup_status") or "unknown").lower()
    expected_minutes = _f(row, "expected_minutes")
    minutes_source = _s(row, "lineup_source")
    sample = _f(row, "rates_sample_matches")
    rates_source = _s(row, "rates_source")
    line = _f(row, "line")

    risk_tags: list[str] = []

    # p_field hierarchy: p_field_est drives the recommendation. crowd_percent
    # is post-lock evaluation only and never feeds the recommendation.
    p_field_est = _f(row, "p_field_est")
    out["p_field_est"] = "" if p_field_est is None else round(p_field_est, 4)

    if qt not in SUPPORTED_QTS:
        out["decision"] = "unsupported_question_type"
        out["review_reason"] = f"question_type {qt!r} not scored by v0"
        out["risk_tags"] = ""
        return out

    # --- bench/availability risk tags (independent of scoring path) ---
    if status == "bench_unknown":
        risk_tags.append("bench_unknown")
    if status == "bench_high_usage":
        risk_tags.append("high_usage_bench_possible")
    if status == "out_of_squad":
        risk_tags.append("out_of_squad")

    # --- factual availability override ---
    if status == "out_of_squad":
        p_truth = 0.0
        out["p_truth_v0"] = 0.0
        out["p_truth_source"] = "availability_fact"
        confidence = "high"  # factual, not a rate estimate
        out["truth_confidence"] = confidence
        _apply_sizing(out, p_truth, p_field_est, confidence, risk_tags)
        return out

    # --- required-input gating ---
    if qt == "player_sot_over":
        required = {"sot_per90": _f(row, "sot_per90")}
    else:  # player_goal_or_assist
        required = {
            "goals_per90": _f(row, "goals_per90"),
            "assists_per90": _f(row, "assists_per90"),
        }
    missing_rates = [k for k, v in required.items() if v is None]

    if expected_minutes is None:
        risk_tags.append("missing_expected_minutes")
    if missing_rates:
        risk_tags.append("missing_rates")

    if expected_minutes is None or missing_rates:
        out["decision"] = "review_missing_inputs"
        bits = []
        if expected_minutes is None:
            bits.append("expected_minutes")
        bits.extend(missing_rates)
        out["review_reason"] = "missing required input(s): " + ", ".join(bits)
        out["truth_confidence"] = "review"
        out["risk_tags"] = ";".join(risk_tags)
        return out

    # --- p_truth_v0 (Poisson) ---
    if qt == "player_sot_over":
        lam = required["sot_per90"] * expected_minutes / 90.0
        k = _line_to_k(line)
        p_truth = poisson_p_ge_k(lam, k)
        out["p_truth_source"] = f"poisson_v0_sot(lambda={lam:.4f},k={k})"
    else:
        lam = (required["goals_per90"] + required["assists_per90"]) * expected_minutes / 90.0
        p_truth = poisson_p_ge_k(lam, 1)
        out["p_truth_source"] = f"poisson_v0_ga(lambda={lam:.4f})"
        # optional xG/xA diagnostic — never replaces p_truth_v0
        xg, xa = _f(row, "xg_per90"), _f(row, "xa_per90")
        if xg is not None and xa is not None:
            lam_x = (xg + xa) * expected_minutes / 90.0
            out["p_truth_xgxa"] = round(poisson_p_ge_k(lam_x, 1), 4)

    out["p_truth_v0"] = round(p_truth, 4)

    # --- thin-sample tag ---
    if (sample or 0) < MEDIUM_SAMPLE:
        risk_tags.append("thin_rate_sample")

    # --- confidence (sizing only) ---
    confidence = confidence_tier(
        status=status,
        expected_minutes=expected_minutes,
        minutes_source=minutes_source,
        sample=sample,
        rates_source=rates_source,
        has_required_rates=True,
    )
    out["truth_confidence"] = confidence
    if confidence == "low":
        risk_tags.append("low_confidence")

    _apply_sizing(out, p_truth, p_field_est, confidence, risk_tags)
    return out


def _apply_sizing(
    out: dict,
    p_truth: float,
    p_field_est: float | None,
    confidence: str,
    risk_tags: list[str],
) -> None:
    """Fill edge / shrink / cap / recommendation / RBP. Mutates ``out``."""
    if p_field_est is None:
        out["decision"] = "scored_no_field"
        out["review_reason"] = (
            "no p_field_est -> p_truth_v0 reported, no recommended submission"
        )
        out["risk_tags"] = ";".join(_dedup(risk_tags))
        return

    edge = p_truth - p_field_est
    out["edge_vs_field"] = round(edge, 4)
    if abs(edge) >= LARGE_EDGE:
        risk_tags.append("large_edge_vs_field")
        risk_tags.append("large_fade" if edge < 0 else "large_chase")

    if confidence == "review":
        out["decision"] = "review"
        out["review_reason"] = "confidence=review; no recommendation"
        out["risk_tags"] = ";".join(_dedup(risk_tags))
        return

    shrink = SHRINK_BY_CONFIDENCE[confidence]
    cap = MAX_DEV_BY_CONFIDENCE[confidence]
    raw = p_field_est + shrink * edge
    capped = max(p_field_est - cap, min(p_field_est + cap, raw))
    rec = round(round(capped * 100) / 100, 2)  # whole percentage points

    out["shrink_used"] = shrink
    out["max_deviation_cap"] = cap
    out["recommended_submission"] = rec
    worst, best = rbp_worst_best(rec, p_field_est)
    out["worst_case_rbp_estimate"] = worst
    out["best_case_rbp_estimate"] = best
    out["decision"] = "scored"
    out["risk_tags"] = ";".join(_dedup(risk_tags))


def _dedup(items: list[str]) -> list[str]:
    seen: list[str] = []
    for x in items:
        if x not in seen:
            seen.append(x)
    return seen


def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = [score_row(r) for r in df.to_dict("records")]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> int:
    fixtures = [
        {  # 1. Ben Waine-style SOT
            "collection_id": "t_waine", "match": "NZ vs Egypt",
            "target_player": "Ben Waine", "target_team": "New Zealand",
            "question_type": "player_sot_over", "line": 0.5,
            "lineup_status": "bench_unknown", "lineup_role": "central_attacker",
            "expected_minutes": 30, "lineup_source": "manager_presser",
            "sot_per90": 0.85, "rates_sample_matches": 8, "rates_source": "fbref",
            "p_field_est": 0.32,
        },
        {  # 2. Trezeguet-style G/A
            "collection_id": "t_trez", "match": "NZ vs Egypt",
            "target_player": "Mahmoud Trezeguet", "target_team": "Egypt",
            "question_type": "player_goal_or_assist",
            "lineup_status": "bench_unknown", "lineup_role": "wide_attacker",
            "expected_minutes": 25, "lineup_source": "beat_writer",
            "goals_per90": 0.22, "assists_per90": 0.18,
            "rates_sample_matches": 8, "rates_source": "fbref",
            "p_field_est": 0.28,
        },
        {  # 3. missing expected_minutes -> review
            "collection_id": "t_nomin", "question_type": "player_sot_over",
            "line": 0.5, "lineup_status": "starter",
            "sot_per90": 1.2, "rates_sample_matches": 12, "rates_source": "fbref",
            "p_field_est": 0.5,
        },
        {  # 4. missing rates -> review
            "collection_id": "t_norate", "question_type": "player_goal_or_assist",
            "lineup_status": "starter", "expected_minutes": 90,
            "rates_sample_matches": 12, "rates_source": "fbref", "p_field_est": 0.4,
        },
        {  # 5. out_of_squad -> p_truth 0, availability_fact
            "collection_id": "t_out", "question_type": "player_sot_over",
            "line": 0.5, "lineup_status": "out_of_squad", "p_field_est": 0.25,
        },
    ]
    df = score_frame(pd.DataFrame(fixtures))
    r = {row["collection_id"]: row for _, row in df.iterrows()}
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + ": " + msg)
        ok = ok and bool(cond)

    w = r["t_waine"]
    check(w["decision"] == "scored", "Waine scored")
    check(0.23 <= w["p_truth_v0"] <= 0.27, f"Waine p_truth_v0 ~0.247 (got {w['p_truth_v0']})")
    check(w["edge_vs_field"] < 0, "Waine edge negative (truth < field)")
    check(w["truth_confidence"] == "medium", f"Waine confidence medium (got {w['truth_confidence']})")
    dev = abs(w["recommended_submission"] - w["p_field_est"])
    check(dev <= w["max_deviation_cap"] + 1e-9, "Waine within deviation cap")
    check(w["recommended_submission"] >= 0.25, f"Waine not nuclear (got {w['recommended_submission']})")

    t = r["t_trez"]
    check(t["decision"] == "scored", "Trezeguet scored")
    check(0.09 <= t["p_truth_v0"] <= 0.12, f"Trez p_truth_v0 ~0.105 (got {t['p_truth_v0']})")
    # partial fade: between p_truth and field, not single-digit
    check(
        t["p_truth_v0"] < t["recommended_submission"] < t["p_field_est"],
        f"Trez partial fade (rec {t['recommended_submission']} between "
        f"{t['p_truth_v0']} and {t['p_field_est']})",
    )
    check(t["recommended_submission"] >= 0.10, f"Trez not single-digit (got {t['recommended_submission']})")

    check(r["t_nomin"]["decision"] == "review_missing_inputs", "missing minutes -> review")
    check(r["t_nomin"]["p_truth_v0"] == "", "missing minutes -> blank p_truth")
    check("missing_expected_minutes" in r["t_nomin"]["risk_tags"], "missing minutes tag")

    check(r["t_norate"]["decision"] == "review_missing_inputs", "missing rates -> review")
    check(r["t_norate"]["p_truth_v0"] == "", "missing rates -> blank p_truth")
    check("missing_rates" in r["t_norate"]["risk_tags"], "missing rates tag")

    o = r["t_out"]
    check(o["p_truth_v0"] == 0.0, "out_of_squad -> p_truth 0")
    check(o["p_truth_source"] == "availability_fact", "out_of_squad -> availability_fact")
    check("out_of_squad" in o["risk_tags"], "out_of_squad tag")

    print("\n" + ("ALL SELFTESTS PASSED" if ok else "SELFTEST FAILURES"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/models/player_prop_collection_template.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/models/player_prop_edge_report.csv"))
    ap.add_argument("--selftest", action="store_true", help="run built-in acceptance fixtures and exit")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")
    df = pd.read_csv(args.input)
    report = score_frame(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)

    n = len(report)
    by_decision = report["decision"].value_counts().to_dict() if n else {}
    print(f"scored {n} row(s) -> {args.out}")
    for k, v in by_decision.items():
        print(f"  {k}: {v}")
    scored = report[report["decision"] == "scored"]
    if len(scored):
        print("\n  recommendations:")
        for _, r in scored.iterrows():
            print(
                f"    {str(r['target_player']):<22} {str(r['question_type']):<22} "
                f"p_truth={r['p_truth_v0']} field={r['p_field_est']} "
                f"-> submit {r['recommended_submission']} "
                f"[{r['truth_confidence']}] tags={r['risk_tags']}"
            )


if __name__ == "__main__":
    main()
