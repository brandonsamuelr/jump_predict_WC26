"""Price tomorrow's player-prop rows from one-sided markets + coverage report.

Reads a question CSV (default: tomorrow's 40), finds the player-prop rows,
prices each against the latest cached per-event odds via the strict
``player_prop_pricing`` handler, and prints:

  1. a player-prop coverage list (covered / weakly covered / unsupported),
  2. a before/after market-p coverage table for ALL question rows.

Does NOT use locked crowd %, does NOT invent No-sides, and never cross-maps
prop types (anytime-scorer is never used for SOT or 2H-SOT).

Example
-------
  python scripts/player_prop_sheet.py \\
      --questions data/submission_sheets/2026-06-23_questions.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib.player_prop_pricing import price_player_prop, PROP_EQUIVALENCE

PROP_TYPES = set(PROP_EQUIVALENCE.keys())


def latest_event_cache(event_id: str) -> Path | None:
    fs = sorted(
        glob.glob(f"data/raw/soccer_fifa_world_cup__event-{event_id}__*.json")
    )
    return Path(fs[-1]) if fs else None


def load_game(event_id: str) -> dict | None:
    cache = latest_event_cache(event_id)
    if cache is None:
        return None
    raw = json.loads(cache.read_text())
    return raw if isinstance(raw, dict) else (raw[0] if raw else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions",
        default="data/submission_sheets/2026-06-23_questions.csv",
    )
    args = ap.parse_args()

    q = pd.read_csv(args.questions, dtype=str).fillna("")
    games: dict[str, dict] = {}
    results: list[dict] = []

    for _, row in q.iterrows():
        qt = row["question_type"].strip().lower()
        rec = {
            "match": row["match"],
            "qnum": row["question_number"],
            "question_type": qt,
            "player": row["target_player"],
            "before": row["before_coverage"],
            "after": row["before_coverage"],  # default: unchanged
            "p_raw": "",
            "p_adj": "",
            "books": "",
            "sharp": "",
            "liq": "",
            "status": "",
        }
        if qt in PROP_TYPES:
            eid = row["event_id"]
            if eid not in games:
                g = load_game(eid)
                if g is not None:
                    games[eid] = g
            game = games.get(eid)
            if game is None:
                rec["status"] = "no_cache"
            else:
                line = float(row["line"]) if row["line"] else None
                pr = price_player_prop(qt, row["target_player"] or None, line, game)
                rec["status"] = pr.status
                if pr.mapped:
                    rec["p_raw"] = f"{pr.market_prob_raw:.3f}"
                    rec["p_adj"] = f"{pr.market_prob_vig_adjusted:.3f}"
                    rec["books"] = str(pr.book_count)
                    rec["sharp"] = str(pr.sharp_book_count)
                    rec["liq"] = pr.liquidity_flag
                    rec["after"] = (
                        "market_prop" if pr.confidence == "direct"
                        else "market_prop_partial"
                    )
                else:
                    rec["after"] = "unsupported"
        results.append(rec)

    res = pd.DataFrame(results)

    # ---- 1. player-prop coverage buckets ----
    props = res[res["question_type"].isin(PROP_TYPES)]
    covered = props[props["after"] == "market_prop"]
    weak = props[props["after"] == "market_prop_partial"]
    unsup = props[props["after"] == "unsupported"]

    print("=" * 78)
    print("PLAYER-PROP COVERAGE (one-sided, vig-adjusted)")
    print("=" * 78)
    print("\n-- COVERED (direct market-equivalent) --")
    if covered.empty:
        print("  (none)")
    for _, r in covered.iterrows():
        print(
            f"  {r['match'][:22]:22s} {r['qnum']:3s} {r['player']:18s} "
            f"raw={r['p_raw']} adj={r['p_adj']} books={r['books']} "
            f"sharp={r['sharp']} [{r['liq']}]"
        )
    print("\n-- WEAKLY COVERED (partial / low-confidence) --")
    if weak.empty:
        print("  (none)")
    for _, r in weak.iterrows():
        print(
            f"  {r['match'][:22]:22s} {r['qnum']:3s} {r['player']:18s} "
            f"raw={r['p_raw']} adj={r['p_adj']} [{r['status']}]"
        )
    print("\n-- UNSUPPORTED (no equivalent market / not quoted) --")
    if unsup.empty:
        print("  (none)")
    for _, r in unsup.iterrows():
        print(
            f"  {r['match'][:22]:22s} {r['qnum']:3s} {r['player']:18s} "
            f"-> {r['status']}"
        )

    # ---- 2. before/after coverage table (all rows) ----
    print("\n" + "=" * 78)
    print("BEFORE / AFTER MARKET-P COVERAGE (all 40)")
    print("=" * 78)
    gained = res[(res["before"] == "needs_model") & (res["after"].str.startswith("market"))]
    print(
        f"rows gaining market p from this build: {len(gained)} "
        f"(player-prop handler)\n"
    )
    show = res[["match", "qnum", "question_type", "before", "after",
                "p_adj", "liq"]].copy()
    show.columns = ["match", "q", "type", "before", "after", "p_adj", "liq"]
    pd.set_option("display.width", 200, "display.max_rows", 60)
    print(show.to_string(index=False))

    print("\nGAINED rows:")
    for _, r in gained.iterrows():
        print(f"  {r['match']} {r['qnum']} {r['player']}: needs_model -> {r['after']} (adj p={r['p_adj']})")


if __name__ == "__main__":
    main()
