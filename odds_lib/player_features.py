"""Player-prop feature extraction (FEATURES ONLY — never p_truth).

This module turns a player-prop question + lineup context + match context
into a flat, structured :class:`PlayerPropFeatures` record. It is the single
source of truth for "what do we know about this player prop before lock".

Hard discipline boundary
-------------------------
Nothing here produces a probability for the event. Lineup status/role,
expected minutes, favourite/underdog, implied goals, etc. are *features* and
*risk tags*. They feed:

  - the manual-review decision path in ``decision_engine`` (risk tags +
    diagnostics), and
  - the offline feature dataset (``scripts/build_player_prop_features.py``)
    that a real, data-derived player-prop model will train on later.

Turning any of these features into a hard-coded ``p_truth`` is explicitly
disallowed (see the model-discipline notes in ``decision_engine.py``).

No import of ``decision_engine`` here — the match context is duck-typed so
this module stays a leaf dependency and avoids an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from .lineups import (
    PlayerContext,
    STARTER_STATUSES,
    BENCH_STATUSES,
    OUT_STATUSES,
)


# Player-prop question types this module knows how to describe. Mirrors
# ``LINEUP_AWARE_QUESTION_TYPES`` in the engine; kept here so the dataset
# builder can run without importing the engine.
PLAYER_PROP_QUESTION_TYPES: set[str] = {
    "player_sot_over",
    "player_sot_2h_over",
    "player_goal_or_assist",
    "player_goal",
}


@dataclass
class PlayerPropFeatures:
    """Pre-lock features for one player-prop question.

    Every field is knowable BEFORE the question locks. Post-lock / outcome
    fields (result, actual_rbp, locked crowd %) deliberately live only in the
    dataset builder's evaluation columns, never here — keeping leakage out of
    the feature object by construction.
    """

    # Identity
    match: str | None = None
    game_date: str | None = None
    question: str | None = None
    question_type: str | None = None
    target_player: str | None = None
    target_team: str | None = None
    line: float | None = None

    # Lineup features
    lineup_status: str = "unknown"
    lineup_role: str = "unknown"
    has_lineup_context: bool = False
    is_starter: bool = False
    is_not_starting: bool = False
    is_out_of_squad: bool = False
    expected_minutes_if_known: float | None = None

    # Match context features
    favorite_team: str | None = None
    underdog_team: str | None = None
    target_team_win_prob: float | None = None
    target_team_is_favorite: bool | None = None
    target_team_is_underdog: bool | None = None
    team_implied_goals_if_available: float | None = None
    match_total_over_2_5_prob: float | None = None
    btts_prob: float | None = None

    # Market / field diagnostics
    has_direct_market: bool = False
    direct_market_prob_if_available: float | None = None
    p_field_est: float | None = None
    p_field_source: str | None = None

    # Derived risk tags (also surfaced on the recommendation)
    risk_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["risk_tags"] = ";".join(self.risk_tags) if self.risk_tags else ""
        return d


# Column order for the offline feature dataset. Evaluation-only columns
# (outcome/locked-crowd) are appended by the builder and intentionally are
# NOT part of PlayerPropFeatures so they can never leak into training X.
FEATURE_COLUMNS: list[str] = [
    "match",
    "game_date",
    "question",
    "question_type",
    "target_player",
    "target_team",
    "line",
    "lineup_status",
    "lineup_role",
    "has_lineup_context",
    "is_starter",
    "is_not_starting",
    "is_out_of_squad",
    "expected_minutes_if_known",
    "favorite_team",
    "underdog_team",
    "target_team_win_prob",
    "target_team_is_favorite",
    "target_team_is_underdog",
    "team_implied_goals_if_available",
    "match_total_over_2_5_prob",
    "btts_prob",
    "has_direct_market",
    "direct_market_prob_if_available",
    "p_field_est",
    "p_field_source",
    "risk_tags",
]

# Evaluation-only columns — for calibration/backtest, NEVER features.
EVAL_ONLY_COLUMNS: list[str] = [
    "submitted_percent",
    "crowd_percent",
    "result",
    "actual_rbp",
]


def classify_lineup_flags(status: str) -> tuple[bool, bool, bool]:
    """Return ``(is_starter, is_not_starting, is_out_of_squad)``.

    ``is_not_starting`` is True for any confirmed bench or out status. It is
    False when status is ``unknown`` (we don't know, so we don't claim it).
    """
    s = (status or "unknown").strip().lower()
    is_starter = s in STARTER_STATUSES
    is_out = s in OUT_STATUSES
    is_not_starting = (s in BENCH_STATUSES) or is_out
    return is_starter, is_not_starting, is_out


def _lineup_status_tag(status: str) -> str | None:
    """Map a lineup status onto the spec risk-tag vocabulary."""
    s = (status or "unknown").strip().lower()
    if s in STARTER_STATUSES:
        return "player_starter"
    if s in OUT_STATUSES:
        return "player_out_of_squad"
    if s in BENCH_STATUSES:
        return "player_not_starting"
    return None


def build_player_prop_features(
    *,
    question_row: dict[str, Any],
    player_ctx: PlayerContext,
    match_ctx: Any = None,
    p_field_est: float | None = None,
    p_field_source: str | None = None,
    has_direct_market: bool = False,
    direct_market_prob: float | None = None,
    game_date: str | None = None,
) -> PlayerPropFeatures:
    """Assemble :class:`PlayerPropFeatures` from the available context.

    ``match_ctx`` is duck-typed (a ``MatchContext`` from the engine, or any
    object exposing the same attribute names). All inputs are pre-lock.
    """
    qt = _clean_str(question_row.get("question_type")) or ""
    qt = qt.lower()
    target_team = _clean_str(question_row.get("target_team"))

    is_starter, is_not_starting, is_out = classify_lineup_flags(player_ctx.status)
    has_lineup_context = (
        player_ctx.status != "unknown" or player_ctx.role != "unknown"
    )

    fav = getattr(match_ctx, "favorite_team", None)
    und = getattr(match_ctx, "underdog_team", None)
    fav_prob = getattr(match_ctx, "favorite_win_prob", None)
    und_prob = getattr(match_ctx, "underdog_win_prob", None)
    p_over = getattr(match_ctx, "p_over", None)
    p_btts = getattr(match_ctx, "p_btts_yes", None)

    target_is_fav: bool | None = None
    target_is_und: bool | None = None
    target_win_prob: float | None = None
    if target_team and fav is not None:
        tnorm = target_team.strip().lower()
        if tnorm == str(fav).strip().lower():
            target_is_fav, target_is_und, target_win_prob = True, False, fav_prob
        elif und is not None and tnorm == str(und).strip().lower():
            target_is_fav, target_is_und, target_win_prob = False, True, und_prob

    feats = PlayerPropFeatures(
        match=_clean_str(question_row.get("match")),
        game_date=game_date,
        question=_clean_str(question_row.get("sports_predict_question")),
        question_type=qt,
        target_player=_clean_str(question_row.get("target_player")),
        target_team=target_team,
        line=_to_float(question_row.get("line")),
        lineup_status=player_ctx.status,
        lineup_role=player_ctx.role,
        has_lineup_context=has_lineup_context,
        is_starter=is_starter,
        is_not_starting=is_not_starting,
        is_out_of_squad=is_out,
        expected_minutes_if_known=player_ctx.expected_minutes,
        favorite_team=fav,
        underdog_team=und,
        target_team_win_prob=target_win_prob,
        target_team_is_favorite=target_is_fav,
        target_team_is_underdog=target_is_und,
        # team_implied_goals is intentionally left None: we have no clean
        # data-derived source yet, and fabricating it would violate the
        # no-hard-coded-numbers discipline. See the model plan doc.
        team_implied_goals_if_available=None,
        match_total_over_2_5_prob=p_over,
        btts_prob=p_btts,
        has_direct_market=has_direct_market,
        direct_market_prob_if_available=direct_market_prob,
        p_field_est=p_field_est,
        p_field_source=p_field_source,
    )
    feats.risk_tags = player_prop_risk_tags(feats)
    return feats


def player_prop_risk_tags(feats: PlayerPropFeatures) -> list[str]:
    """Risk tags for a player prop, following the spec vocabulary.

    These describe *what is known/unknown*; they never imply a probability.
    """
    tags: list[str] = ["player_prop"]

    if feats.has_lineup_context:
        tags.append("lineup_context_available")
        status_tag = _lineup_status_tag(feats.lineup_status)
        if status_tag:
            tags.append(status_tag)
        if feats.lineup_role != "unknown":
            tags.append(f"lineup_role:{feats.lineup_role}")
        if feats.expected_minutes_if_known is None and not feats.is_starter:
            tags.append("expected_minutes_unknown")
    else:
        tags.append("missing_lineup_context")

    if not feats.has_direct_market:
        tags.append("no_direct_market")
    # No trained/empirical player-prop model exists yet anywhere in the
    # pipeline, so this is always true for player props today. Kept explicit
    # so the day a model lands, the absence/presence is visible in the data.
    tags.append("no_trained_player_prop_model")

    return tags


def _clean_str(value: Any) -> str | None:
    """Coerce a value (incl. pandas NaN floats) to a trimmed string or None."""
    if value is None:
        return None
    s = str(value).strip()
    if s in ("", "nan", "None", "NaN"):
        return None
    return s


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s in ("", "nan", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


__all__ = [
    "PlayerPropFeatures",
    "PLAYER_PROP_QUESTION_TYPES",
    "FEATURE_COLUMNS",
    "EVAL_ONLY_COLUMNS",
    "build_player_prop_features",
    "player_prop_risk_tags",
    "classify_lineup_flags",
]
