"""Deliverable 2 — override leakage report.

Quantifies whether manual overrides helped or hurt, by category. For every
resolved row where the locked value departed from the pipeline value, the leak
is the PAIRED RBP difference:

    leak = rbp_final - rbp_pipeline      (< 0 => the override cost us points)

Aggregated by override category (see measurement.classify_override). This is
the evidence behind the default policy: SOFT overrides (human distrust of a
confident model / pulling trusted rows to the middle / vague game-script
intuition) are DISABLED by default in apply_override; HARD_QA factual fixes are
kept; everything earns its place by measured leakage, not vibes.

    python scripts/override_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from odds_lib.measurement import LOG_PATH, score_rows, classify_override

DEV_EPS = 0.011   # |final - pipeline| above this is a real override (not rounding)


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def main():
    raw = pd.read_csv(LOG_PATH, dtype=str)
    scored = score_rows(raw).set_index(["match", "question_number"])

    rows = []
    for _, r in raw.iterrows():
        pipe = _num(r.get("pipeline_submit"))
        final = _num(r.get("final_submitted"))
        reason = str(r.get("override_reason") or "")
        cat = classify_override(reason)
        dev = (final - pipe) if (pipe is not None and final is not None) else None
        is_override = dev is not None and abs(dev) > DEV_EPS
        key = (r["match"], r["question_number"])
        sc = scored.loc[key] if key in scored.index else None
        leak = (float(sc["rbp_final"]) - float(sc["rbp_pipeline"])) if sc is not None else None
        rows.append({
            "match": r["match"], "q": r["question_number"], "type": r["question_type"],
            "category": cat, "is_override": is_override,
            "pipeline": pipe, "final": final,
            "dev": round(dev, 3) if dev is not None else None,
            "resolved": sc is not None,
            "leak": round(leak, 2) if leak is not None else None,
            "reason": reason[:60],
        })
    df = pd.DataFrame(rows)

    ov = df[df["is_override"]].copy()           # real overrides only (not rounding)
    ov_res = ov[ov["resolved"] & ov["leak"].notna()]

    print("=" * 100)
    print("DELIVERABLE 2 — OVERRIDE LEAKAGE  (leak = rbp_final - rbp_pipeline; <0 = override hurt)")
    print("=" * 100)
    print(f"{len(df)} logged rows | {len(ov)} real overrides (|dev|>{DEV_EPS}) | "
          f"{len(ov_res)} of those resolved & scorable")

    if not ov_res.empty:
        print("\n--- leakage by category (resolved overrides) ---")
        agg = ov_res.groupby("category")["leak"].agg(["count", "mean", "sum"]).round(2)
        agg = agg.rename(columns={"count": "n", "mean": "mean_leak", "sum": "total_leak"})
        print(agg.to_string())
        tot_n, tot_leak = len(ov_res), ov_res["leak"].sum()
        print(f"\nALL resolved overrides: n={tot_n}  mean_leak={tot_leak/tot_n:+.2f}/q  "
              f"total={tot_leak:+.1f} RBP")
        soft = ov_res[ov_res["category"] == "soft"]
        if not soft.empty:
            print(f"  SOFT subset (the disabled category): n={len(soft)}  "
                  f"mean_leak={soft['leak'].mean():+.2f}/q  total={soft['leak'].sum():+.1f} RBP")
        hard = ov_res[ov_res["category"] == "hard_qa"]
        if not hard.empty:
            print(f"  HARD_QA subset (kept): n={len(hard)}  "
                  f"mean_leak={hard['leak'].mean():+.2f}/q  total={hard['leak'].sum():+.1f} RBP")

    print("\n--- every real override (audit the category labels) ---")
    pd.set_option("display.width", 200, "display.max_rows", 60, "display.max_colwidth", 62)
    show = ov[["match", "q", "type", "category", "pipeline", "final", "dev", "leak", "reason"]]
    print(show.to_string(index=False))

    n_pending = int((ov["resolved"] == False).sum())  # noqa: E712
    print(f"\nNOTE: {n_pending} real overrides are still UNRESOLVED — their leak is not yet known.")
    print("POLICY (now enforced in apply_override): SOFT category disabled by default")
    print("  (raise allow_soft=True for a deliberate, logged exception); HARD_QA kept; re-run")
    print("  this report as rows resolve so each category earns its place by measured leakage.")
    print("CAVEAT: leak is correlated within a match (one game script) — read total across")
    print(f"  matches ({ov_res['match'].nunique() if not ov_res.empty else 0} clusters), not as independent rows.")


if __name__ == "__main__":
    main()
