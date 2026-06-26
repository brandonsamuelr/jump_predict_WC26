"""Generate a SportsPredict submission sheet from question_inventory + cached odds.

This is the daily driver for SportsPredict. It maps each curated question
(via data/question_inventory.csv) to a sportsbook market, computes the
consensus probability, and recommends submit / review / skip per row.

The odds layer combines two API endpoints:

- bulk ``/odds`` for h2h, totals (one call covers all upcoming events)
- per-event ``/events/{id}/odds`` for markets the bulk endpoint does not
  surface for soccer (btts, h2h_h1, etc.) — one call per event per market

Per-event markets are governed by a minimum-book-count gate so a single
obscure book can't drive a submission. Below ``--min-books`` we route to
``low_liquidity_review`` (no auto-submit). Between ``--min-books`` and
``--thin-books-threshold`` we still allow ``mapped_exact`` but flag
``liquidity_flag = thin`` so the user reviews the line manually.

Reads
-----
- data/question_inventory.csv
- the latest cached bulk + per-event odds (or fresh with --fetch)

Writes
------
- data/submission_sheets/{forecast_run_id}_submit_sheet.csv
- data/predictions_log.csv (only the rows recommended to submit) when --log

Prints
------
A focused review table sorted by kickoff and a recommendation summary.

Examples
--------
  # Build a sheet using the latest caches (no API calls):
  python scripts/submit_sheet.py --forecast-run-id 2026-06-21_morning

  # Fetch fresh bulk + per-event odds and log:
  python scripts/submit_sheet.py --forecast-run-id 2026-06-21_pregame \\
      --fetch --log
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from odds_lib import (
    append_csv,
    append_prelock_rows,
    CALIBRATION_LOG_PATH,
    fetch_event_odds,
    fetch_odds,
    json_to_markets,
    latest_bulk_cache,
    latest_event_cache,
    load_lineup,
    load_priors,
    map_question,
    process_match,
    recommend_submission,
)
from odds_lib.odds import DEFAULT_SHARP_BOOKS

INVENTORY_PATH = Path("data/question_inventory.csv")
OUTPUT_DIR = Path("data/submission_sheets")
PREDICTIONS_LOG_PATH = Path("data/predictions_log.csv")

DIAGNOSTIC_COLS = [
    "market_prob_all",
    "all_num_books",
    "all_books_used",
    "all_std_prob",
    "all_min_prob",
    "all_max_prob",
    "all_range_prob",
    "market_prob_sharp",
    "sharp_num_books",
    "sharp_books_used",
    "sharp_std_prob",
    "sharp_min_prob",
    "sharp_max_prob",
    "sharp_range_prob",
    "abs_diff_all_vs_sharp",
]

DECISION_COLS = [
    "p_truth",
    "p_truth_source",
    "truth_confidence",
    "p_field",
    "p_field_source",
    "field_confidence",
    "p_submit",
    "decision_mode",
    "needs_manual_review",
    "delta_vs_field",
    "estimated_swing",
    "risk_tags",
    "reason",
    "historical_candidate",
    "candidate_n",
    "candidate_raw_bias",
    "candidate_reason",
    "promotion_status",
]

SHEET_COLS = [
    "kickoff_time_local",
    "match",
    "sports_predict_question",
    "question_type",
    "target_team",
    "mapped_market",
    "mapped_line",
    "mapped_outcome",
    "mapped_bet_description",
    "market_prob",
    "submit_prob",
    "submit_percent",
    *DECISION_COLS,
    *DIAGNOSTIC_COLS,
    "mapping_status",
    "liquidity_flag",
    "review_flags",
    "submit_recommendation",
    "manual_override",
    "source",
    "notes",
]

LOG_COLS = [
    "forecast_run_id",
    "submitted_at",
    "match",
    "kickoff_time_utc",
    "sports_predict_question",
    "question_type",
    "target_team",
    "mapped_market",
    "mapped_outcome",
    "mapped_line",
    "mapped_bet_description",
    "market_prob",
    "submit_prob",
    "submit_percent",
    "field_prob",
    *DECISION_COLS,
    *DIAGNOSTIC_COLS,
    "liquidity_flag",
    "review_flags",
    "strategy_name",
    "mapping_status",
    "manual_override",
    "source_cache",
    "per_event_caches",
    "source",
    "notes",
]

# Disagreement gate thresholds. Pulled from spec.
STD_DISAGREEMENT_THRESHOLD = 0.06
RANGE_DISAGREEMENT_THRESHOLD = 0.15
SHARP_NOTE_THRESHOLD = 0.02
SHARP_FORCE_THRESHOLD = 0.04


# Decision modes the engine produces that are safe to auto-submit when
# include=1. Default-to-submit: any defensible field estimate qualifies.
# ``weak_field`` and ``review`` never auto-submit — those rows are surfaced
# for manual decision regardless of include.
AUTO_SUBMITTABLE_MODES = {
    "direct_market",
    "derived_market",
    "strong_historical",
    "lean",
    "contextual_shadow_with_bias_hint",
    "contextual_shadow",
}


def _recommendation(mapping_status: str, include: int, decision_mode: str) -> str:
    """Map (mapping_status, include, decision_mode) onto sheet-level recommendation.

    The contest is a chase — shadow rows have positive expected RBP via the
    Jensen gap on field Brier variance. Skipping = 0. So we default to submit
    whenever the engine produced a defensible field estimate. Only
    ``weak_field`` / ``review`` rows are withheld from auto-submit.
    """
    if decision_mode in AUTO_SUBMITTABLE_MODES:
        return "submit" if include == 1 else "review_not_submit"
    if decision_mode in ("weak_field", "review", "player_prop_review_required"):
        return "review"
    if mapping_status in ("ambiguous_review", "low_liquidity_review"):
        return "review"
    return "skip"


def _compute_review(
    mp, include: int, manual_override: int, decision_mode: str
) -> tuple[str, list[str]]:
    """Apply the all-vs-sharp disagreement rules on top of the base recommendation.

    Returns ``(recommendation, review_flags)``. Rules (cumulative):

    - ``liquidity_flag == thin`` AND ``sharp_num_books == 0`` -> force
      ``review_not_submit`` (always; not overridable).
    - ``all_std_prob >= 0.06`` or ``all_range_prob >= 0.15`` -> tag
      ``high_disagreement_review`` (does not change the recommendation).
    - ``abs_diff_all_vs_sharp >= 0.02`` -> tag ``sharp_disagreement``.
    - ``abs_diff_all_vs_sharp >= 0.04`` -> force ``review_not_submit``
      unless ``manual_override == 1``; either way the
      ``sharp_disagreement`` tag stays.
    """
    rec = _recommendation(mp.mapping_status, include, decision_mode)
    flags: list[str] = []

    if mp.mapping_status == "mapped_exact":
        if mp.liquidity_flag == "thin" and (mp.sharp_num_books or 0) == 0:
            rec = "review_not_submit"
            flags.append("thin_no_sharp_review")

        std = mp.all_std_prob
        rng = mp.all_range_prob
        if (std is not None and std >= STD_DISAGREEMENT_THRESHOLD) or (
            rng is not None and rng >= RANGE_DISAGREEMENT_THRESHOLD
        ):
            flags.append("high_disagreement_review")

        diff = mp.abs_diff_all_vs_sharp
        if diff is not None and diff >= SHARP_NOTE_THRESHOLD:
            flags.append("sharp_disagreement")
        if diff is not None and diff >= SHARP_FORCE_THRESHOLD:
            if manual_override == 1:
                flags.append("sharp_disagreement_override")
            else:
                rec = "review_not_submit"

    return rec, flags


def _parse_line(value) -> float | None:
    s = str(value).strip()
    if s in ("", "nan", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _resolve_caches(
    args: argparse.Namespace, in_window: list[dict], per_event_market_keys: tuple[str, ...]
) -> dict[str, Path]:
    """Return ``{event_id: cache_path}`` for per-event odds.

    With ``--fetch``: makes one per-event API call per (event, market). With
    cache only: looks up the newest matching per-event cache per event_id;
    events with no cache are silently absent from the dict (the mapper will
    return ``needs_model`` for any per-event question on those matches).
    """
    if not per_event_market_keys:
        return {}

    per_event_markets = ",".join(per_event_market_keys)
    out: dict[str, Path] = {}

    if args.fetch:
        n_events = len(in_window)
        if n_events == 0:
            return out
        cost = (
            n_events
            * len(per_event_market_keys)
            * len(args.regions.split(","))
        )
        print(
            f"\nfetching per-event markets [{per_event_markets}] for "
            f"{n_events} event(s) — est. {cost} credits..."
        )
        for game in in_window:
            event_id = game["id"]
            label = f"{game['home_team']} vs {game['away_team']}"
            try:
                cache = fetch_event_odds(
                    args.sport,
                    event_id,
                    markets=per_event_markets,
                    regions=args.regions,
                )
                out[event_id] = cache
            except requests.HTTPError as e:
                print(f"  WARN: per-event fetch failed for {label} ({event_id}): {e}")
        return out

    for game in in_window:
        cache = latest_event_cache(args.sport, game["id"], args.regions)
        if cache is not None:
            out[game["id"]] = cache
    print(
        f"  per-event caches available: {len(out)}/{len(in_window)} "
        f"(markets requested: {per_event_markets})"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build a SportsPredict submission sheet.")
    # DEPRECATED / QUARANTINED 2026-06-26. This script builds lock sheets from the OLDER
    # decision_engine path (p_field-anchored blend with +/-max_dev caps), NOT the guarded
    # optimize() path. The optimize() path has the TRUST_PRICE_K + universal-guard protections
    # that make it structurally impossible to shrink a market read toward c_hat; this one does
    # NOT carry those guarantees. LOCK FROM scripts/pregame_refresh.py (per match) or
    # scripts/submission_optimizer.py (full slate) instead. Refuses to run without an explicit
    # acknowledgement flag so nobody accidentally locks from the unguarded engine.
    p.add_argument("--use-unguarded-decision-engine", action="store_true",
                   help="REQUIRED to run: acknowledges this is the UNGUARDED engine (see pregame_refresh).")
    p.add_argument("--forecast-run-id", required=True)
    p.add_argument("--sport", default="soccer_fifa_world_cup")
    p.add_argument(
        "--markets",
        default="h2h,totals",
        help="Comma-separated bulk-endpoint market keys (h2h, totals, ...).",
    )
    p.add_argument(
        "--per-event-markets",
        default="btts,h2h_h1",
        help="Comma-separated per-event market keys. Empty string to disable.",
    )
    p.add_argument("--regions", default="us,uk,eu")
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--local-tz", default="America/New_York")
    p.add_argument(
        "--min-books",
        type=int,
        default=3,
        help="Below this num_books a mapped market is downgraded to "
        "low_liquidity_review (not auto-submitted).",
    )
    p.add_argument(
        "--thin-books-threshold",
        type=int,
        default=5,
        help="Between --min-books and this threshold, mapped_exact stays but "
        "liquidity_flag is set to 'thin' so the user reviews manually.",
    )
    p.add_argument(
        "--sharp-books",
        default=",".join(DEFAULT_SHARP_BOOKS),
        help=(
            "Comma-separated bookmaker titles to treat as the sharp basket "
            "for the sharp_* diagnostics. Default: "
            + ", ".join(DEFAULT_SHARP_BOOKS)
        ),
    )
    p.add_argument(
        "--fetch",
        action="store_true",
        help="Call the API (costs credits) instead of using the latest caches.",
    )
    p.add_argument(
        "--log",
        action="store_true",
        help="Append the submit-recommended rows to data/predictions_log.csv.",
    )
    p.add_argument(
        "--strategy",
        default="market_consensus_v1",
        help="Strategy name recorded with each logged row.",
    )
    args = p.parse_args()

    if not args.use_unguarded_decision_engine:
        print(
            "REFUSING TO RUN — submit_sheet.py uses the UNGUARDED decision_engine path.\n"
            "  A market read here can be blended toward the field (the failure that overrode the\n"
            "  Turkiye-win line). Lock from the GUARDED optimize() path instead:\n"
            "    • per match : python scripts/pregame_refresh.py --match \"...\"\n"
            "    • full slate: python scripts/submission_optimizer.py\n"
            "  If you truly need this legacy sheet, re-run with --use-unguarded-decision-engine.",
            file=sys.stderr,
        )
        sys.exit(2)

    if not INVENTORY_PATH.exists():
        print(
            f"FATAL: {INVENTORY_PATH} not found. Run scripts/stub_inventory.py first, "
            "or create the file with just the header row."
        )
        sys.exit(1)

    inventory = pd.read_csv(INVENTORY_PATH).fillna("")
    inventory["include"] = (
        pd.to_numeric(inventory["include"], errors="coerce").fillna(0).astype(int)
    )
    # manual_override is opt-in: set to 1 on a row to bypass the
    # abs_diff_all_vs_sharp >= 0.04 forcing. Default 0 / absent column.
    if "manual_override" not in inventory.columns:
        inventory["manual_override"] = 0
    inventory["manual_override"] = (
        pd.to_numeric(inventory["manual_override"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    sharp_books = tuple(
        b.strip() for b in args.sharp_books.split(",") if b.strip()
    )

    # ---- bulk odds ----
    if args.fetch:
        bulk_cache = fetch_odds(args.sport, markets=args.markets, regions=args.regions)
    else:
        bulk_cache = latest_bulk_cache(args.sport, args.regions)
        print(f"using bulk cache: {bulk_cache}")

    raw = json.loads(Path(bulk_cache).read_text())
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=args.hours)
    in_window = [
        g
        for g in raw
        if now
        <= datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00"))
        <= cutoff
    ]
    print(f"games kicking off within next {args.hours}h: {len(in_window)}")
    if not in_window:
        print("no upcoming games; nothing to do.")
        return

    kickoff_by_match = {
        f"{g['home_team']} vs {g['away_team']}": g["commence_time"] for g in in_window
    }
    in_window_matches = set(kickoff_by_match)

    bulk_market_keys = tuple(m.strip() for m in args.markets.split(",") if m.strip())
    per_event_market_keys = tuple(
        m.strip() for m in args.per_event_markets.split(",") if m.strip()
    )

    # ---- per-event odds ----
    per_event_caches = _resolve_caches(args, in_window, per_event_market_keys)

    bulk_markets = json_to_markets(
        in_window,
        forecast_run_id=args.forecast_run_id,
        market_keys=bulk_market_keys,
    )
    per_event_frames = []
    for event_id, cache in per_event_caches.items():
        per_event_frames.append(
            json_to_markets(
                cache,
                forecast_run_id=args.forecast_run_id,
                market_keys=per_event_market_keys,
            )
        )

    if per_event_frames:
        all_markets = pd.concat([bulk_markets, *per_event_frames], ignore_index=True)
    else:
        all_markets = bulk_markets

    _, consensus, _ = process_match(
        all_markets, on_incomplete="drop", sharp_books=sharp_books
    )

    inv_window = inventory[inventory["match"].isin(in_window_matches)].copy()

    # ---- warnings ----
    matches_with_any_inventory = set(inv_window["match"].unique())
    matches_with_include = set(
        inv_window[inv_window["include"] == 1]["match"].unique()
    )
    no_inventory_at_all = in_window_matches - matches_with_any_inventory
    inventory_but_no_include = (
        in_window_matches - matches_with_include - no_inventory_at_all
    )
    if no_inventory_at_all:
        print(
            f"\nWARNING: {len(no_inventory_at_all)} match(es) in odds window but "
            "no inventory rows at all:"
        )
        for m in sorted(no_inventory_at_all):
            print(f"  - {m}")
        print("  -> run scripts/stub_inventory.py or add rows manually.")
    if inventory_but_no_include:
        print(
            f"\nNote: {len(inventory_but_no_include)} match(es) have inventory rows "
            "but none marked include=1:"
        )
        for m in sorted(inventory_but_no_include):
            print(f"  - {m}")

    if inv_window.empty:
        print("\nno inventory rows match the upcoming games; exiting.")
        return

    # ---- build sheet ----
    local_tz = ZoneInfo(args.local_tz)
    priors = load_priors()
    # Pre-load per-match lineups (silently missing for matches without files).
    lineups_by_match: dict[str, object] = {}
    for m in in_window_matches:
        lu = load_lineup(m)
        if lu is not None:
            lineups_by_match[m] = lu
    if lineups_by_match:
        print(
            f"loaded {len(lineups_by_match)} lineup file(s): "
            + ", ".join(sorted(lineups_by_match))
        )
    rows = []
    for _, irow in inv_window.iterrows():
        match = irow["match"]
        sub_consensus = consensus[consensus["match"] == match]
        mp = map_question(
            question_type=str(irow.get("question_type", "")),
            target_team=str(irow.get("target_team", "")) or None,
            target_player=str(irow.get("target_player", "")) or None,
            line=_parse_line(irow.get("line", "")),
            consensus=sub_consensus,
            min_books=args.min_books,
            thin_books_threshold=args.thin_books_threshold,
        )
        lineup_ctx = (
            {"lineup": lineups_by_match[match]} if match in lineups_by_match else None
        )
        decision = recommend_submission(
            question_row=irow.to_dict(),
            market_context={"mapping": mp, "consensus": sub_consensus},
            historical_context=priors,
            lineup_context=lineup_ctx,
        )
        kickoff_utc = pd.Timestamp(kickoff_by_match[match], tz="UTC")
        kickoff_local = kickoff_utc.tz_convert(local_tz)

        # Direct markets: submit_prob still tracks market_prob (engine sets
        # p_submit = market_prob in that mode). Fallback rows now carry
        # engine-derived submit_prob so the sheet has a value to surface.
        market_prob = mp.market_prob
        submit_prob = decision.p_submit if decision.p_submit is not None else market_prob
        submit_percent = (
            int(round(submit_prob * 100)) if submit_prob is not None else None
        )
        manual_override = int(irow["manual_override"])
        rec, flags = _compute_review(
            mp, int(irow["include"]), manual_override, decision.decision_mode
        )

        def _v(x):
            return x if x is not None else ""

        def _f3(x):
            return round(float(x), 4) if x is not None else ""

        rows.append(
            {
                "kickoff_time_local": kickoff_local.strftime("%Y-%m-%d %H:%M %Z"),
                "kickoff_time_utc": kickoff_utc.isoformat(),
                "match": match,
                "sports_predict_question": irow.get("sports_predict_question", ""),
                "question_type": irow.get("question_type", ""),
                "target_team": irow.get("target_team", ""),
                "mapped_market": mp.mapped_market or "",
                "mapped_line": mp.mapped_line if mp.mapped_line is not None else "",
                "mapped_outcome": mp.mapped_outcome or "",
                "mapped_bet_description": mp.mapped_bet_description or "",
                "market_prob": _v(market_prob),
                "submit_prob": _v(submit_prob),
                "submit_percent": _v(submit_percent),
                "p_truth": _f3(decision.p_truth),
                "p_truth_source": decision.p_truth_source,
                "truth_confidence": _f3(decision.truth_confidence),
                "p_field": _f3(decision.p_field),
                "p_field_source": decision.p_field_source,
                "field_confidence": _f3(decision.field_confidence),
                "p_submit": _f3(decision.p_submit),
                "decision_mode": decision.decision_mode,
                "needs_manual_review": decision.needs_manual_review,
                "delta_vs_field": _f3(decision.delta_vs_field),
                "estimated_swing": _f3(decision.estimated_swing),
                "risk_tags": ";".join(decision.risk_tags),
                "reason": decision.reason,
                "historical_candidate": decision.historical_candidate,
                "candidate_n": decision.candidate_n,
                "candidate_raw_bias": _f3(decision.candidate_raw_bias),
                "candidate_reason": decision.candidate_reason,
                "promotion_status": decision.promotion_status,
                "market_prob_all": _v(mp.market_prob_all),
                "all_num_books": _v(mp.all_num_books),
                "all_books_used": mp.all_books_used,
                "all_std_prob": _v(mp.all_std_prob),
                "all_min_prob": _v(mp.all_min_prob),
                "all_max_prob": _v(mp.all_max_prob),
                "all_range_prob": _v(mp.all_range_prob),
                "market_prob_sharp": _v(mp.market_prob_sharp),
                "sharp_num_books": mp.sharp_num_books,
                "sharp_books_used": mp.sharp_books_used,
                "sharp_std_prob": _v(mp.sharp_std_prob),
                "sharp_min_prob": _v(mp.sharp_min_prob),
                "sharp_max_prob": _v(mp.sharp_max_prob),
                "sharp_range_prob": _v(mp.sharp_range_prob),
                "abs_diff_all_vs_sharp": _v(mp.abs_diff_all_vs_sharp),
                "mapping_status": mp.mapping_status,
                "liquidity_flag": mp.liquidity_flag,
                "review_flags": ";".join(flags),
                "submit_recommendation": rec,
                "manual_override": manual_override,
                "source": irow.get("source", ""),
                "notes": irow.get("notes", ""),
            }
        )

    sheet = pd.DataFrame(rows)
    sheet = sheet.sort_values(
        ["kickoff_time_utc", "match", "submit_recommendation"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.forecast_run_id}_submit_sheet.csv"
    sheet[SHEET_COLS].to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    # ---- terminal view ----
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== submission sheet (review) ===")
    display = sheet.copy()
    # Compact numeric display so the terminal view stays readable.
    for col in (
        "market_prob_sharp",
        "abs_diff_all_vs_sharp",
        "all_std_prob",
        "all_range_prob",
    ):
        display[col] = display[col].apply(
            lambda v: f"{float(v):.3f}" if v not in ("", None) else ""
        )
    print(
        display[
            [
                "kickoff_time_local",
                "match",
                "sports_predict_question",
                "submit_percent",
                "all_num_books",
                "sharp_num_books",
                "market_prob_sharp",
                "abs_diff_all_vs_sharp",
                "all_std_prob",
                "all_range_prob",
                "liquidity_flag",
                "review_flags",
                "submit_recommendation",
            ]
        ].to_string(index=False)
    )

    print("\n=== summary by submit_recommendation ===")
    print(sheet["submit_recommendation"].value_counts().to_string())

    # ---- log ----
    if not args.log:
        print("\n--log not set; predictions_log.csv unchanged")
        return

    to_log = sheet[sheet["submit_recommendation"] == "submit"].copy()
    if to_log.empty:
        print("\n--log set but no rows recommend submit; nothing logged.")
        return

    per_event_summary = ";".join(
        f"{eid}:{Path(p).name}" for eid, p in per_event_caches.items()
    )
    to_log["submitted_at"] = pd.Timestamp.now(tz="UTC")
    to_log["forecast_run_id"] = args.forecast_run_id
    to_log["strategy_name"] = args.strategy
    to_log["source_cache"] = str(bulk_cache)
    to_log["per_event_caches"] = per_event_summary
    append_csv(to_log[LOG_COLS], PREDICTIONS_LOG_PATH)
    print(f"\nappended {len(to_log)} row(s) to {PREDICTIONS_LOG_PATH}")

    # Calibration log — schema-compatible with sportspredict_submitted_scoring_rows.csv
    # so backfilled post-lock rows can be merged with historical for analysis.
    n_cal = append_prelock_rows(sheet, args.forecast_run_id, CALIBRATION_LOG_PATH)
    print(f"appended {n_cal} pre-lock row(s) to {CALIBRATION_LOG_PATH}")


if __name__ == "__main__":
    main()
