"""Deliverable 6 — forward leaderboard pace tracker.

Static gaps ("we're 857 behind #100") don't tell us if the strategy is WORKING.
What matters is the FORWARD rate: per slate, are we out-scoring #100 (closing
the gap) or not? This reads leaderboard snapshots and, between consecutive ones,
computes forward RBP-per-weighted-question for us / #100 / #1, the net gap
change, and — at the current pace — whether and when we close.

The per-period denominator is OUR weighted-question delta (sum of multipliers of
newly resolved questions). The contest is max-volume on a shared slate, so #100
and #1 face ~the same weighted-q per period; we use the common denominator to
compare rates. (Documented assumption — we cannot see rivals' question counts.)

    python scripts/pace_tracker.py
    python scripts/pace_tracker.py --add --date 2026-06-24 --label fri_slate \\
        --field 3239 --rank 640 --our 175 --r100 1030 --r1 1525 [--wq 36]

If --wq is omitted on --add, the weighted-q to date is inferred from resolved
rows in the measurement log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib.measurement import LOG_PATH, score_rows

SNAP_PATH = Path("data/leaderboard_snapshots.csv")
REMAINING_MATCHES = 60   # ~matches left (handoff); for the required-rate target
Q_PER_MATCH = 10         # ~scoring questions per match


def _infer_weighted_q() -> int | None:
    if not LOG_PATH.exists():
        return None
    scored = score_rows(pd.read_csv(LOG_PATH, dtype=str))
    if scored.empty:
        return None
    return int(scored["multiplier"].fillna(1).sum())


def add_snapshot(args):
    wq = args.wq if args.wq is not None else _infer_weighted_q()
    row = {
        "date": args.date, "slate_label": args.label, "field_size": args.field,
        "our_rank": args.rank, "our_rbp": args.our, "rbp_rank100": args.r100,
        "rbp_rank1": args.r1, "our_weighted_q_cum": wq if wq is not None else "",
        "notes": args.notes or "",
    }
    df = pd.read_csv(SNAP_PATH) if SNAP_PATH.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(SNAP_PATH, index=False)
    print(f"appended snapshot {args.date} ({args.label}); weighted_q_cum={row['our_weighted_q_cum']}")


def report():
    df = pd.read_csv(SNAP_PATH).sort_values("date").reset_index(drop=True)
    last = df.iloc[-1]
    gap100 = last["rbp_rank100"] - last["our_rbp"]
    gap1 = last["rbp_rank1"] - last["our_rbp"]
    rem_wq = REMAINING_MATCHES * Q_PER_MATCH

    print("=" * 92)
    print("DELIVERABLE 6 — FORWARD PACE TRACKER")
    print("=" * 92)
    print(f"latest ({last['date']}, {last['slate_label']}): rank {int(last['our_rank'])} of "
          f"{int(last['field_size'])} | our_rbp={last['our_rbp']:.0f} | "
          f"#100={last['rbp_rank100']:.0f} | #1={last['rbp_rank1']:.0f}")
    print(f"static gaps: to #100 = {gap100:.0f} RBP | to #1 = {gap1:.0f} RBP")
    print(f"to close #100 over ~{REMAINING_MATCHES} matches (~{rem_wq} weighted-q) we must OUT-score")
    print(f"  #100 by {gap100/rem_wq:+.2f} RBP/weighted-q (and #1 by {gap1/rem_wq:+.2f}).")

    if len(df) < 2:
        print("\nonly the baseline snapshot exists — forward pace appears after the next")
        print("snapshot. Add one after the next slate resolves:")
        print("  python scripts/pace_tracker.py --add --date <d> --label <l> --field <n> "
              "--rank <r> --our <rbp> --r100 <rbp> --r1 <rbp>")
        return

    print("\n--- forward pace per period (Δ between snapshots; rate per OUR weighted-q) ---")
    out = []
    for i in range(1, len(df)):
        a, b = df.iloc[i - 1], df.iloc[i]
        dq = (b["our_weighted_q_cum"] - a["our_weighted_q_cum"]) if pd.notna(a["our_weighted_q_cum"]) and pd.notna(b["our_weighted_q_cum"]) else float("nan")
        dq = dq if (dq == dq and dq > 0) else float("nan")
        d_our = b["our_rbp"] - a["our_rbp"]
        d100 = b["rbp_rank100"] - a["rbp_rank100"]
        d1 = b["rbp_rank1"] - a["rbp_rank1"]
        g100_prev, g100_now = a["rbp_rank100"] - a["our_rbp"], b["rbp_rank100"] - b["our_rbp"]
        rate = (lambda x: round(x / dq, 2) if dq == dq else float("nan"))
        out.append({
            "period": f"{a['date']}→{b['date']}", "dq": dq if dq == dq else "?",
            "our/q": rate(d_our), "#100/q": rate(d100), "#1/q": rate(d1),
            "edge_vs_100/q": rate(d_our - d100), "edge_vs_1/q": rate(d_our - d1),
            "Δgap100": round(g100_now - g100_prev, 1),  # <0 = we CLOSED on #100
            "gap100": round(g100_now, 1),
        })
    rep = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(rep.to_string(index=False))

    # projection from the most recent period
    last_edge = rep.iloc[-1]["edge_vs_100/q"]
    print()
    if isinstance(last_edge, float) and last_edge == last_edge and last_edge > 0:
        close_wq = gap100 / last_edge
        print(f"at the latest edge_vs_100 ({last_edge:+.2f}/weighted-q), the #100 gap closes in "
              f"~{close_wq:.0f} weighted-q (~{close_wq/Q_PER_MATCH:.0f} matches).")
        verdict = "ON TRACK" if close_wq <= rem_wq else "TOO SLOW (gap closes after the contest ends)"
        print(f"  remaining budget ~{rem_wq} weighted-q -> {verdict}.")
    else:
        print("latest period: we are NOT out-scoring #100 (edge_vs_100 <= 0) -> the gap is NOT")
        print("  closing at this pace. The strategy must lift RBP/weighted-q above #100's rate.")
    print("\nCAVEAT: #100/#1 per-q rates assume a shared max-volume slate (common weighted-q).")
    print("One period is noisy — read the trend across several slates, not a single row.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="store_true", help="append a snapshot then report")
    ap.add_argument("--date"); ap.add_argument("--label", default="")
    ap.add_argument("--field", type=int); ap.add_argument("--rank", type=int)
    ap.add_argument("--our", type=float); ap.add_argument("--r100", type=float)
    ap.add_argument("--r1", type=float); ap.add_argument("--wq", type=float, default=None)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    if args.add:
        missing = [k for k in ("date", "field", "rank", "our", "r100", "r1")
                   if getattr(args, k) is None]
        if missing:
            ap.error(f"--add requires: {', '.join('--' + m for m in missing)}")
        add_snapshot(args)
        print()
    report()


if __name__ == "__main__":
    main()
