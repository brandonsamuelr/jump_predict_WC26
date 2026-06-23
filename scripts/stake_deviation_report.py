"""Deliverable 4 — stake × deviation slate report (pre-lock soundness QA).

Surfaces the rows where we are taking the LARGEST position against the field, so
a human can sanity-check the INPUTS (is the player starting? line right? mapping
correct? odds stale? class assignment right?). It is deliberately NOT a
moderation tool: the output is a QA checklist, not an instruction to shrink. A
big, well-founded deviation is the point — we just want to be sure it's well
founded before it locks.

Per row, on the edge-weighted submit sheet:
    raw_claim = p_hat  - c_hat     (how far the MODEL departs from the field)
    expressed = SUBMIT - c_hat     (how far we ACTUALLY submit, after k)
    priority  = |expressed| * stake   (stake = question multiplier, default 1)

Two flags:
  * BIG POSITION  — large |expressed|*stake: most exposed if the model is wrong.
  * SHRUNK CLAIM  — large |raw_claim| but k pulled it in hard (|expressed| <<
    |raw_claim|): confirm the class/k is right (are we under-expressing a real
    edge, or correctly distrusting a thin one?).

    python scripts/stake_deviation_report.py [--sheet <csv>] [--top 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

DEFAULT_SHEET = "data/submission_sheets/2026-06-23_optimized_submit_sheet.csv"
BIG_EXPRESSED = 0.08   # |submit - c_hat| above this is a notable position
SHRINK_RATIO = 0.5     # expressed < this * raw_claim => the claim was pulled in hard


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=DEFAULT_SHEET)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv(args.sheet)
    # tolerate both the new (c_hat) and older (shadow) column name for the proxy
    chat_col = "c_hat" if "c_hat" in df.columns else "shadow"
    has_mult = "multiplier" in df.columns

    rows = []
    for _, r in df.iterrows():
        c = _num(r.get(chat_col))
        sub = _num(r.get("SUBMIT"))
        pm = _num(r.get("p_hat"))
        if c is None or sub is None:
            continue
        stake = (_num(r.get("multiplier")) or 1.0) if has_mult else 1.0
        raw = (pm - c) if pm is not None else float("nan")
        expr = sub - c
        rows.append({
            "match": r.get("match"), "q": r.get("q"), "type": r.get("type"),
            "class": r.get("class", r.get("tier", "")), "k": _num(r.get("k")),
            "c_hat": round(c, 3), "p_hat": round(pm, 3) if pm is not None else "",
            "SUBMIT": round(sub, 3),
            "raw_claim": round(raw, 3) if raw == raw else "",
            "expressed": round(expr, 3),
            "stake": stake,
            "priority": round(abs(expr) * stake, 3),
        })
    rep = pd.DataFrame(rows)
    if rep.empty:
        print("no scorable rows in sheet."); return

    pd.set_option("display.width", 200, "display.max_colwidth", 32)
    print("=" * 96)
    print("DELIVERABLE 4 — STAKE × DEVIATION (pre-lock soundness QA, NOT moderation)")
    print("=" * 96)
    print(f"sheet: {args.sheet}  |  c_hat from '{chat_col}'  |  "
          f"stake = {'multiplier' if has_mult else 'flat 1 (no multiplier column)'}")

    big = rep.sort_values("priority", ascending=False).head(args.top)
    print(f"\n--- TOP {len(big)} BIG POSITIONS (largest |submit - c_hat| × stake) — QA the inputs ---")
    print(big[["match", "q", "type", "class", "k", "c_hat", "p_hat", "SUBMIT",
               "expressed", "stake", "priority"]].to_string(index=False))

    shrunk = rep[rep.apply(
        lambda x: (x["raw_claim"] != "" and abs(x["raw_claim"]) >= BIG_EXPRESSED
                   and abs(x["expressed"]) < SHRINK_RATIO * abs(x["raw_claim"])), axis=1)]
    shrunk = shrunk.sort_values("raw_claim", key=lambda s: s.abs(), ascending=False)
    print(f"\n--- SHRUNK CLAIMS (big model view, k pulled it in >50%) — confirm class/k is right ---")
    if shrunk.empty:
        print("  none.")
    else:
        print(shrunk[["match", "q", "type", "class", "k", "c_hat", "p_hat",
                      "SUBMIT", "raw_claim", "expressed"]].to_string(index=False))

    print(f"\nSoundness checklist for each BIG POSITION (do NOT moderate on vibes):")
    print("  player starting & role/minutes confirmed? line/threshold correct? question")
    print("  mapping correct? odds fresh (not stale)? class/k assignment right? If all pass,")
    print("  LOCK the deviation — that is the edge. Only a FAILED check (hard-QA) changes a row.")


if __name__ == "__main__":
    main()
