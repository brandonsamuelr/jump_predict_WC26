"""Settlement-scope classifier + scope guard for knockout winner-like markets.

In knockouts, sportsbook NAMING is dangerous: "Team A to win" can be a regulation 3-way, an
ET+pens to-advance market, a draw-no-bet, or a 2-way winner -- each settles differently. This
module classifies a market by SETTLEMENT (from its outcome labels + title), and gates which scope
may feed which row. It is deliberately NOT a connector: it protects manually/semi-manually entered
external lines, and the existing feed, from scope mismatches.

The load-bearing rule: the REGULATION_3WAY market is the strength signal for the whole slate
(favorite-gap -> team goals, BTTS, totals splits, HT lead, first goal, corner/card comparisons,
prop context). NEVER let an ET+pens ADVANCE market feed a regulation-scoped row -- that leaks
extra-time/penalty mass into questions that resolve in regulation. And NEVER price an advance row
off a 90' regulation market, nor off a GROUP_STAGE_QUALIFICATION market (the "advance to knockout
stages" trap = group -> R32, which is NOT "advance to R16" = win the R32 tie).
"""
from __future__ import annotations

SCOPE_CLASSES = (
    "REGULATION_3WAY",            # 90'+stoppage, Draw is an outcome. Strength signal for the slate.
    "ADVANCE_ET_PENS",            # includes ET+penalties, no draw. Valid ONLY for advance rows.
    "DRAW_NO_BET",                # draw pushes/refunds. Not valid raw for win-in-reg or advance.
    "TWO_WAY_WINNER_UNKNOWN",     # 2 outcomes, no draw, unclear settlement -> never use blindly.
    "GROUP_STAGE_QUALIFICATION",  # "advance to knockout stages" = group->R32. NOT win-the-tie.
    "OUTRIGHT_FUTURE",            # tournament winner / futures. Not a specific tie.
    "UNSUPPORTED",                # scope cannot be verified.
)


def classify_market(outcome_labels, title: str = "", note: str = "") -> str:
    """Classify a winner-like market by settlement. Structure wins where it is unambiguous
    (a 3-way with Draw IS the regulation match market, even under a 'Team A to win' title);
    title text disambiguates the no-draw cases and flags the known traps."""
    labels = {str(l).strip().lower() for l in (outcome_labels or []) if str(l).strip()}
    txt = (str(title) + " " + str(note)).lower()
    has_draw = "draw" in labels or "tie" in labels
    n = len(labels)

    # explicit traps first (override structure)
    if any(k in txt for k in ("knockout stage", "knockout stages", "group stage",
                              "qualify from group", "reach the knockout", "out of the group")):
        return "GROUP_STAGE_QUALIFICATION"
    if any(k in txt for k in ("outright", "tournament winner", "win the world cup",
                              "to lift", "futures", "winner of the tournament")):
        return "OUTRIGHT_FUTURE"
    if "draw no bet" in txt or "dnb" in txt:
        return "DRAW_NO_BET"

    # structural: a 3-way with a Draw outcome is unambiguously the regulation match market
    if has_draw and n >= 3:
        return "REGULATION_3WAY"

    # no draw, two outcomes: advance if the title says so, else ambiguous (never use blindly)
    if not has_draw and n == 2:
        if any(k in txt for k in ("to advance", "advance to", "to qualify", "to reach",
                                  "reach round", "win the tie", "to progress", "progress to")):
            return "ADVANCE_ET_PENS"
        return "TWO_WAY_WINNER_UNKNOWN"
    return "UNSUPPORTED"


# which market scopes may legitimately source which row scope
_VALID = {
    "regulation": {"REGULATION_3WAY"},          # win-in-reg, regulation tie, favorite-gap, all reg rows
    "advance":    {"ADVANCE_ET_PENS"},          # advance-to-R16 only (settlement-verified)
}


def valid_source_for(desired_scope: str, market_scope: str) -> tuple[bool, str]:
    """Gate: may a market of ``market_scope`` price a row whose settlement is ``desired_scope``
    ('regulation' | 'advance')? Returns (ok, reason). Encodes the no-leak rules."""
    allowed = _VALID.get(desired_scope)
    if allowed is None:
        return False, f"unknown desired_scope {desired_scope!r} (expected 'regulation' or 'advance')"
    if market_scope in allowed:
        return True, "ok"
    if desired_scope == "regulation":
        return False, (f"regulation row needs REGULATION_3WAY; {market_scope} would leak ET/pens "
                       "or mis-settle (ADVANCE/DNB/2-way are wrong scope)")
    return False, (f"advance row needs settlement-verified ADVANCE_ET_PENS; {market_scope} is wrong "
                   "scope (REGULATION_3WAY omits ET/pens; GROUP_STAGE_QUALIFICATION = group->R32, "
                   "NOT win-the-tie)")


# --- HARD CONTAINMENT: an external (ET+pens) advance price is quarantined to the advance row ----
# It may replace ONLY an ADVANCE_MARKET row. It must NEVER update regulation favorite_gap, team
# lambdas, corners, cards, player props, or any regulation-scoped route -- doing so would leak
# extra-time/penalty mass into questions that resolve in regulation.
EXTERNAL_ADVANCE_ALLOWED_ROUTES = frozenset({"ADVANCE_MARKET"})
REGULATION_PROTECTED = ("favorite_gap", "team_lambdas", "corners", "cards", "player_props",
                        "regulation_3way", "totals", "btts", "halftime", "first_goal")


def can_apply_external_advance(route_class: str) -> bool:
    """True iff an external advance (ET+pens) price may update this route. ADVANCE_MARKET only."""
    return route_class in EXTERNAL_ADVANCE_ALLOWED_ROUTES


def assert_external_advance_target(route_class: str) -> None:
    """Raise unless ``route_class`` may receive an external advance price (the hard rule)."""
    if not can_apply_external_advance(route_class):
        raise ValueError(
            f"external advance (ET+pens) price may NOT update a {route_class!r} route -- it is "
            f"quarantined to {sorted(EXTERNAL_ADVANCE_ALLOWED_ROUTES)} and must never touch "
            f"regulation-scoped quantities {REGULATION_PROTECTED}")


# --- orientation: read home/away from the FEED, never from a routing label ----------------------
def _norm(s: str) -> str:
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z ]", " ", s).split())


def feed_orientation(game_json: dict) -> tuple[str | None, str | None]:
    """(home_team, away_team) as the FEED reports them. The single source of truth for orientation;
    routing labels are display only (CAN-RSA + Morocco-NL are flipped vs their labels)."""
    return game_json.get("home_team"), game_json.get("away_team")


def target_in_feed(game_json: dict, team: str) -> str | None:
    """The feed team NAME matching ``team`` (orientation-independent), or None if absent. The explicit
    lock-time check: a routing-label team MUST exist in the feed (None => stale/flipped/wrong label,
    do not price by position)."""
    for t in feed_orientation(game_json):
        if t and _norm(t) == _norm(team):
            return t
    for t in feed_orientation(game_json):
        if t and (_norm(team) in _norm(t) or _norm(t) in _norm(team)):
            return t
    return None


__all__ = ["SCOPE_CLASSES", "classify_market", "valid_source_for",
           "EXTERNAL_ADVANCE_ALLOWED_ROUTES", "REGULATION_PROTECTED",
           "can_apply_external_advance", "assert_external_advance_target",
           "feed_orientation", "target_in_feed"]
