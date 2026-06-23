"""Task 1 — c_hat reliability audit (DIAGNOSTIC ONLY; changes no submission).

Measures how well our PRE-LOCK field proxy c_hat (the ``shadow`` column)
predicted the REALIZED contest crowd c_star (the ``crowd_prob`` column, revealed
only after lock). error = c_hat - c_star. This says whether the Portugal-Q5
stale-anchor miss was a one-off or a systematic per-category failure.

    python scripts/crowd_reliability_report.py

NOTE on units: c_hat/shadow are decimals (0-1); crowd_prob is stored as an
integer percent (e.g. 35). Both are normalized to fractions here. Realized crowd
is POST-LOCK — used here only as a backward-looking benchmark, never as input.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from odds_lib.measurement import LOG_PATH


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _as_fraction(crowd) -> float | None:
    """crowd_prob may be a percent (35) or a fraction (0.35) -> fraction."""
    v = _num(crowd)
    if v is None:
        return None
    return v / 100.0 if v > 1.0 else v


def crowd_error(c_hat, crowd_prob) -> float | None:
    """error = c_hat - realized_crowd, both as fractions. None if unscorable."""
    ch, cs = _num(c_hat), _as_fraction(crowd_prob)
    if ch is None or cs is None:
        return None
    return ch - cs


def bucket(tier: str) -> str:
    t = (tier or "").strip().upper()
    if t == "MARKET":
        return "MARKET"
    if t == "ENGINE_GOALS":
        return "ENGINE"
    if t in ("RATE_SOT", "RATE_SOT_CMP"):
        return "RATE_SOT"
    if t in ("PROP_OK", "PROP_THIN"):
        return "PROP"
    return "PENDING/SHADOW"


def prep(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with both a c_hat (shadow) and a realized crowd, with error columns."""
    rows = []
    for _, r in df.iterrows():
        err = crowd_error(r.get("shadow"), r.get("crowd_prob"))
        if err is None:
            continue
        rows.append({
            "match": r.get("match"), "question_number": r.get("question_number"),
            "question_type": r.get("question_type"), "tier": r.get("tier"),
            "bucket": bucket(r.get("tier")),
            "c_hat": _num(r.get("shadow")), "realized_crowd": _as_fraction(r.get("crowd_prob")),
            "submitted_prob": _num(r.get("final_submitted")),
            "error": err, "abs_error": abs(err),
            "result": r.get("result"), "actual_rbp": _num(r.get("actual_rbp")),
        })
    return pd.DataFrame(rows)


def reliability_table(p: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per-group reliability stats. ALWAYS carries n_rows AND n_match_clusters."""
    if p.empty:
        return pd.DataFrame()
    def agg(g):
        e, ae = g["error"].to_numpy(), g["abs_error"].to_numpy()
        return pd.Series({
            "n_rows": len(g),
            "n_match_clusters": g["match"].nunique(),
            "mean_error": round(float(e.mean()), 3),
            "median_error": round(float(np.median(e)), 3),
            "mean_abs_error": round(float(ae.mean()), 3),
            "rmse": round(float(np.sqrt((e ** 2).mean())), 3),
            "p25_abs": round(float(np.percentile(ae, 25)), 3),
            "p75_abs": round(float(np.percentile(ae, 75)), 3),
            "max_abs_error": round(float(ae.max()), 3),
        })
    out = p.groupby(by).apply(agg, include_groups=False)
    out.loc["ALL"] = agg(p)
    return out


def main():
    df = pd.read_csv(LOG_PATH, dtype=str)
    p = prep(df)
    if p.empty:
        print("no rows with both c_hat and realized crowd yet."); return
    pd.set_option("display.width", 200, "display.max_rows", 80, "display.max_colwidth", 46)

    print("=" * 100)
    print("TASK 1 — c_hat (pre-lock) vs REALIZED crowd reliability  (DIAGNOSTIC ONLY)")
    print("=" * 100)
    print(f"{len(p)} rows with a realized crowd, across {p['match'].nunique()} match clusters. "
          f"error = c_hat - realized_crowd (both fractions).")

    for by, label in [("bucket", "SOURCE BUCKET"), ("tier", "TIER"), ("question_type", "QUESTION_TYPE")]:
        print(f"\n--- by {label} ---")
        print(reliability_table(p, by).to_string())

    print("\n--- WORST INDIVIDUAL MISSES (|error| desc) ---")
    qtext = _load_question_text()
    worst = p.sort_values("abs_error", ascending=False).head(12).copy()
    worst["question"] = [qtext.get((m, q), "") for m, q in zip(worst["match"], worst["question_number"])]
    cols = ["match", "question_number", "question_type", "tier", "submitted_prob",
            "c_hat", "realized_crowd", "error", "abs_error", "result", "actual_rbp"]
    print(worst[cols].to_string(index=False))

    print("\nCLUSTER CAVEAT: crowd errors are correlated WITHIN a match and within a question family,")
    print(f"  so a per-category number from few clusters ({p['match'].nunique()} total) is SUGGESTIVE, not conclusive.")
    print("  Diagnostic only — changes no submission. Use mean_abs_error as the pre-lock stake proxy for Task 2.")


def _load_question_text() -> dict:
    out = {}
    for f in Path("data/submission_sheets").glob("*questions*.csv"):
        try:
            q = pd.read_csv(f, dtype=str).fillna("")
            for _, r in q.iterrows():
                out[(r.get("match"), r.get("question_number"))] = r.get("question", "")[:50]
        except Exception:
            continue
    return out


if __name__ == "__main__":
    main()
