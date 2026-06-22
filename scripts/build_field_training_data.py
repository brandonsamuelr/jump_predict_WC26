"""Build the supervised training dataset for the p_field model.

Reads ``data/historical/sportspredict_collected_data.csv``, extracts
features per question, and writes ``data/models/field_training_rows.csv``.

Label
-----
``y_field = field_prob / 100`` — the locked SportsPredict crowd YES %.
The result and RBP columns are kept as METADATA so audits can join on
them, but they MUST NOT be used as model features (post-prediction leakage).

Features (best-effort)
----------------------
- Question structure indicators (player/team/match prop, compound, halves,
  thresholds, topic flags for fouls/cards/sot/corners/offsides/goals/penalties).
- Threshold line when present.
- Market context (favorite, win probs, totals, BTTS, halftime draw) from
  cached odds when available; missing for almost all historical rows.
  ``has_market_context = 0`` for rows where the market couldn't be matched.
- Role bucket (favorite/underdog/neutral) derived from market context.

Rows are NEVER dropped for missing market context — the trainer handles
imputation downstream. We do drop rows with no resolved field_prob (no
label).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from odds_lib import json_to_markets, process_match, extract_match_context


COLLECTED_PATH = Path("data/historical/sportspredict_collected_data.csv")
OUT_PATH = Path("data/models/field_training_rows.csv")
RAW_DIR = Path("data/raw")


# Cheap topic flags from the question text. Used as binary features so the
# model can pool across question_types with the same underlying topic.
_TOPIC_PATTERNS: list[tuple[str, str]] = [
    ("is_fouls_question",    r"\bfoul"),
    ("is_cards_question",    r"\bcard|\bbooking"),
    ("is_sot_question",      r"\bshots? on target|\bSOT\b"),
    ("is_corners_question",  r"\bcorner"),
    ("is_offsides_question", r"\boffside"),
    ("is_goal_question",     r"\bgoal|\bscore"),
    ("is_penalty_question",  r"\bpenalty"),
]

_TOPIC_RE = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in _TOPIC_PATTERNS]


def _structural_flags(qt: str, question: str) -> dict[str, int]:
    qt_l = (qt or "").lower()
    q_l = (question or "").lower()
    flags = {
        "is_player_prop": int(qt_l.startswith("player_")),
        "is_team_prop": int(qt_l.startswith("team_") or qt_l.startswith("halftime_team_")),
        "is_match_prop": int(
            not qt_l.startswith("player_") and not qt_l.startswith("team_")
            and not qt_l.startswith("halftime_team_")
        ),
        "is_compound_question": int(qt_l.startswith("compound_") or "and" in q_l and (
            "both teams" in q_l or "score and" in q_l or "and have" in q_l
        )),
        "is_second_half_question": int(
            "_2h" in qt_l or "second_half" in qt_l or "second half" in q_l or " 2h" in q_l
        ),
        "is_halftime_question": int(qt_l.startswith("halftime") or "halftime" in q_l or "at half" in q_l),
        "is_threshold_question": int(
            "_over" in qt_l or "_under" in qt_l or any(s in q_l for s in (" or more", " or less", "at least"))
        ),
        "is_favorite_dominance_question": int(qt_l in {
            "team_win", "halftime_team_lead", "halftime_team_winning"
        }),
        "is_underdog_activity_question": int(qt_l in {
            "team_score_2h", "team_score_any", "player_goal", "player_goal_or_assist",
            "team_sot_over", "team_sot_2h_over", "player_sot_over", "player_sot_2h_over"
        }),
    }
    for name, rx in _TOPIC_RE:
        flags[name] = int(bool(rx.search(question or "")))
    return flags


def _parse_line(raw) -> float | None:
    s = str(raw).strip() if raw is not None else ""
    if s in ("", "nan", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Market context lookup
# ---------------------------------------------------------------------------

# Filename pattern: soccer_fifa_world_cup__h2h-totals__us-uk-eu__{ts}.json
_BULK_RE = re.compile(r"^[a-z_]+__h2h-totals__[a-z\-]+__\d{8}T\d{6}Z\.json$")
_EVENT_RE = re.compile(r"^[a-z_]+__event-([a-f0-9]+)__[a-z0-9\-_]+__[a-z\-]+__\d{8}T\d{6}Z\.json$")


def _load_all_bulk_caches() -> list[dict]:
    """Pre-load every cached bulk JSON. Returns a list of (path, raw_events)."""
    out = []
    for p in sorted(RAW_DIR.glob("*.json")):
        if not _BULK_RE.match(p.name):
            continue
        try:
            data = json.loads(p.read_text())
            if isinstance(data, list):
                out.append((p, data))
        except Exception:
            continue
    return out


def _index_event_caches() -> dict[str, Path]:
    """{event_id: newest per-event cache path}."""
    by_id: dict[str, list[Path]] = {}
    for p in sorted(RAW_DIR.glob("*.json")):
        m = _EVENT_RE.match(p.name)
        if m:
            by_id.setdefault(m.group(1), []).append(p)
    return {eid: paths[-1] for eid, paths in by_id.items()}


def _find_event_for_match(
    match_name: str, bulk_caches: list[tuple[Path, list[dict]]]
) -> tuple[str | None, list[dict] | None]:
    """Return (event_id, [event_dict]) for the most-recent cache containing
    a game whose ``home vs away`` matches ``match_name`` (case-insensitive,
    accent-insensitive substring tolerance via lower())."""
    target = match_name.strip().lower()
    # Walk newest-first so we get the freshest snapshot.
    for path, events in reversed(bulk_caches):
        for g in events:
            name = f"{g.get('home_team', '')} vs {g.get('away_team', '')}".lower()
            if name == target:
                return g["id"], [g]
    return None, None


def _build_consensus_for_match(
    match_name: str,
    bulk_caches: list[tuple[Path, list[dict]]],
    event_caches: dict[str, Path],
) -> pd.DataFrame | None:
    event_id, in_window = _find_event_for_match(match_name, bulk_caches)
    if event_id is None:
        return None
    try:
        bm = json_to_markets(in_window, forecast_run_id="train", market_keys=("h2h", "totals"))
        frames = [bm]
        if event_id in event_caches:
            try:
                event_data = json.loads(event_caches[event_id].read_text())
                em = json_to_markets(event_data, forecast_run_id="train", market_keys=("btts", "h2h_h1"))
                frames.append(em)
            except Exception:
                pass
        markets = pd.concat(frames, ignore_index=True)
        _, consensus, _ = process_match(markets, on_incomplete="drop")
        sub = consensus[consensus["match"] == match_name]
        return sub if not sub.empty else None
    except Exception:
        return None


def _market_features(
    match_name: str,
    target_team: str | None,
    consensus_cache: dict[str, pd.DataFrame | None],
    bulk_caches: list[tuple[Path, list[dict]]],
    event_caches: dict[str, Path],
) -> dict:
    """Return market-derived features (favorite, win probs, totals, BTTS, role).

    Cached by match_name so we don't re-process the same consensus per
    question. ``target_team`` only affects the role label and target_team_win_prob.
    """
    if match_name not in consensus_cache:
        consensus_cache[match_name] = _build_consensus_for_match(
            match_name, bulk_caches, event_caches
        )
    sub = consensus_cache[match_name]
    if sub is None or sub.empty:
        return {
            "has_market_context": 0,
            "favorite_team": "",
            "underdog_team": "",
            "favorite_win_prob": None,
            "underdog_win_prob": None,
            "favorite_gap": None,
            "target_team_win_prob": None,
            "target_is_favorite": 0,
            "target_is_underdog": 0,
            "target_is_neutral_or_unknown": 1,
            "target_role_bucket": "unknown",
            "match_total_over_2_5_prob": None,
            "btts_prob": None,
            "halftime_draw_prob": None,
            "target_team_implied_strength": None,
        }
    ctx = extract_match_context(sub)
    t = (target_team or "").strip().lower()
    fav_l = (ctx.favorite_team or "").strip().lower()
    und_l = (ctx.underdog_team or "").strip().lower()
    if t and t == fav_l:
        target_p, role = ctx.favorite_win_prob, "favorite"
    elif t and t == und_l:
        target_p, role = ctx.underdog_win_prob, "underdog"
    else:
        target_p, role = None, "neutral_or_unknown"
    # Collapse close matches to neutral (matches V2 classify_team_role gate).
    if ctx.fav_underdog_gap is not None and ctx.fav_underdog_gap < 0.20:
        role = "neutral_or_unknown"
        if t and (t == fav_l or t == und_l):
            target_p = ctx.favorite_win_prob if t == fav_l else ctx.underdog_win_prob
    return {
        "has_market_context": 1,
        "favorite_team": ctx.favorite_team or "",
        "underdog_team": ctx.underdog_team or "",
        "favorite_win_prob": ctx.favorite_win_prob,
        "underdog_win_prob": ctx.underdog_win_prob,
        "favorite_gap": ctx.fav_underdog_gap,
        "target_team_win_prob": target_p,
        "target_is_favorite": int(role == "favorite"),
        "target_is_underdog": int(role == "underdog"),
        "target_is_neutral_or_unknown": int(role == "neutral_or_unknown"),
        "target_role_bucket": role,
        "match_total_over_2_5_prob": ctx.p_over,
        "btts_prob": ctx.p_btts_yes,
        "halftime_draw_prob": ctx.p_halftime_draw,
        "target_team_implied_strength": target_p,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METADATA_COLS = [
    "match", "match_norm", "game_date", "question_number", "question",
    "question_type", "target_team", "target_player", "line",
    "crowd_percent", "submitted_percent", "result", "actual_rbp", "source_batch",
]

FEATURE_COLS = [
    "has_market_context",
    "favorite_team", "underdog_team",
    "favorite_win_prob", "underdog_win_prob", "favorite_gap",
    "target_team_win_prob", "target_is_favorite", "target_is_underdog",
    "target_is_neutral_or_unknown", "target_role_bucket",
    "match_total_over_2_5_prob", "btts_prob", "halftime_draw_prob",
    "target_team_implied_strength",
    "is_player_prop", "is_team_prop", "is_match_prop",
    "is_compound_question", "is_second_half_question", "is_halftime_question",
    "is_threshold_question", "threshold_line",
    "is_fouls_question", "is_cards_question", "is_sot_question",
    "is_corners_question", "is_offsides_question", "is_goal_question",
    "is_penalty_question", "is_favorite_dominance_question",
    "is_underdog_activity_question", "target_player_present",
]


def main() -> None:
    df = pd.read_csv(COLLECTED_PATH)
    # Need a label.
    df = df[pd.to_numeric(df["field_prob"], errors="coerce").notna()].copy()
    df["crowd_percent"] = pd.to_numeric(df["field_prob"], errors="coerce")
    df = df[df["crowd_percent"].between(0, 100)].copy()
    df["y_field"] = df["crowd_percent"] / 100.0
    if "target_team" not in df.columns:
        df["target_team"] = ""
    if "target_player" not in df.columns:
        df["target_player"] = ""
    if "line" not in df.columns:
        df["line"] = ""

    bulk_caches = _load_all_bulk_caches()
    event_caches = _index_event_caches()
    print(f"loaded {len(bulk_caches)} bulk caches; {len(event_caches)} per-event caches")

    consensus_cache: dict[str, pd.DataFrame | None] = {}
    out_rows = []
    n_market_ctx = 0
    for _, r in df.iterrows():
        match = str(r.get("match_norm") or r.get("match_raw") or "").strip()
        tt = str(r.get("target_team") or "").strip()
        tp = str(r.get("target_player") or "").strip()
        line_val = _parse_line(r.get("line"))
        qt = str(r.get("question_type") or "")
        question = str(r.get("question") or "")

        flags = _structural_flags(qt, question)
        flags["threshold_line"] = line_val
        flags["target_player_present"] = int(bool(tp))

        market = _market_features(match, tt or None, consensus_cache, bulk_caches, event_caches)
        if market["has_market_context"]:
            n_market_ctx += 1

        row = {
            "match": match,
            "match_norm": match,
            "game_date": r.get("game_date", ""),
            "question_number": r.get("question_number", ""),
            "question": question,
            "question_type": qt,
            "target_team": tt,
            "target_player": tp,
            "line": line_val if line_val is not None else "",
            "crowd_percent": r["crowd_percent"],
            "submitted_percent": r.get("submitted_percent", ""),
            "result": r.get("result", ""),
            "actual_rbp": r.get("actual_rbp", ""),
            "source_batch": r.get("source_batch", ""),
            "y_field": r["y_field"],
        }
        row.update(market)
        row.update(flags)
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_cols = ["y_field"] + METADATA_COLS + FEATURE_COLS
    out_df = out_df[[c for c in out_cols if c in out_df.columns]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}: {len(out_df)} rows, "
          f"{n_market_ctx} with market context ({100*n_market_ctx/len(out_df):.1f}%)")
    print(f"label stats: y_field mean={out_df['y_field'].mean():.3f}, "
          f"std={out_df['y_field'].std():.3f}, "
          f"min={out_df['y_field'].min():.3f}, max={out_df['y_field'].max():.3f}")
    print(f"matches: {out_df['match_norm'].nunique()}")
    print(f"question_types: {out_df['question_type'].nunique()}")


if __name__ == "__main__":
    main()
