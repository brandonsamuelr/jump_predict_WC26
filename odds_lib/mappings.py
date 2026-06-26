"""Map SportsPredict question types to sportsbook markets.

Each mapper queries the consensus DataFrame (multi-market output of
``process_match``) for the correct ``(market_key, line, outcome)`` slice. If
the data isn't there we return ``needs_model`` with a clear bet description
explaining what was missing — never a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# question_types we know about but cannot resolve from the API directly yet.
NEEDS_MODEL = {
    "team_corners_over",
    "total_cards_over",
    "team_score_2h",
    "team_score_first",
    "player_sot_over",
    "second_half_more_goals_than_first",
    "halftime_both_teams_sot_1plus",
    "compound_prop",
}

# question_types with no clean market or model path.
UNMAPPED_SKIP = {
    "team_offsides_over",
    "team_more_fouls",
    "team_more_cards",
    "team_more_sot_2h",
    "penalty_or_red_card",
}

# Candidate market keys for first-half / halftime result. We probe in order
# and use whichever the API actually populates.
HALFTIME_H2H_KEYS = ("h2h_h1", "halftime", "first_half_winner")


@dataclass
class Mapping:
    mapped_market: str | None
    mapped_outcome: str | None
    mapped_line: float | None
    mapped_bet_description: str | None
    market_prob: float | None  # pinned to market_prob_all for now
    mapping_status: str  # mapped_exact | needs_model | unmapped_skip | ambiguous_review | low_liquidity_review
    liquidity_flag: str = ""  # "" | ok | thin | low | n/a
    # All-books diagnostics
    market_prob_all: float | None = None
    all_num_books: int | None = None
    all_books_used: str = ""
    all_std_prob: float | None = None
    all_min_prob: float | None = None
    all_max_prob: float | None = None
    all_range_prob: float | None = None
    # Sharp-books diagnostics
    market_prob_sharp: float | None = None
    sharp_num_books: int = 0
    sharp_books_used: str = ""
    sharp_std_prob: float | None = None
    sharp_min_prob: float | None = None
    sharp_max_prob: float | None = None
    sharp_range_prob: float | None = None
    # Comparison
    abs_diff_all_vs_sharp: float | None = None


def _empty(status: str, line: float | None = None) -> Mapping:
    return Mapping(
        mapped_market=None,
        mapped_outcome=None,
        mapped_line=line,
        mapped_bet_description=None,
        market_prob=None,
        mapping_status=status,
        liquidity_flag="n/a",
    )


def _no_market(market_key: str, want: str, line: float | None = None) -> Mapping:
    return Mapping(
        mapped_market=market_key,
        mapped_outcome=want,
        mapped_line=line,
        mapped_bet_description=(
            f"{market_key} / {want} — no consensus row in current cache "
            f"(fetch this market or add a model)"
        ),
        market_prob=None,
        mapping_status="needs_model",
        liquidity_flag="n/a",
    )


def _slice_market(
    consensus: pd.DataFrame, market_key: str, line: float | None = None
) -> pd.DataFrame:
    df = consensus[consensus["market_key"] == market_key]
    if line is None:
        return df[df["line"].isna()]
    return df[~df["line"].isna() & (df["line"] == line)]


def _f(r: pd.Series, col: str) -> float | None:
    """Read a numeric column out of a consensus row, returning None for NaN."""
    v = r.get(col)
    if v is None or pd.isna(v):
        return None
    return float(v)


def _row_to_mapping(
    r: pd.Series,
    market_key: str,
    outcome: str,
    line: float | None,
    description: str,
) -> Mapping:
    sharp_n_raw = r.get("sharp_num_books", 0)
    sharp_n = int(sharp_n_raw) if not pd.isna(sharp_n_raw) else 0
    return Mapping(
        mapped_market=market_key,
        mapped_outcome=outcome,
        mapped_line=line,
        mapped_bet_description=description,
        market_prob=float(r["market_prob"]),
        mapping_status="mapped_exact",
        liquidity_flag="",  # filled in by _apply_liquidity_gate
        market_prob_all=_f(r, "market_prob_all"),
        all_num_books=int(r["all_num_books"]),
        all_books_used=str(r.get("all_books_used", "")),
        all_std_prob=_f(r, "all_std_prob"),
        all_min_prob=_f(r, "all_min_prob"),
        all_max_prob=_f(r, "all_max_prob"),
        all_range_prob=_f(r, "all_range_prob"),
        market_prob_sharp=_f(r, "market_prob_sharp"),
        sharp_num_books=sharp_n,
        sharp_books_used=str(r.get("sharp_books_used", "")),
        sharp_std_prob=_f(r, "sharp_std_prob"),
        sharp_min_prob=_f(r, "sharp_min_prob"),
        sharp_max_prob=_f(r, "sharp_max_prob"),
        sharp_range_prob=_f(r, "sharp_range_prob"),
        abs_diff_all_vs_sharp=_f(r, "abs_diff_all_vs_sharp"),
    )


def _apply_liquidity_gate(
    mp: Mapping, min_books: int, thin_books_threshold: int
) -> Mapping:
    """Post-process a mapping with a per-event-friendly liquidity check.

    UNIFIED PRINCIPLE: book COUNT never triggers a fallback/downgrade. A present market
    (mapped_exact) is a real read regardless of count, so liquidity_flag is now a
    DIAGNOSTIC LABEL only and ``mapping_status`` is NEVER downgraded on count:
    - ``num_books >= thin_books_threshold`` -> "ok"
    - ``1 <= num_books < thin_books_threshold`` -> "thin" (USED, not blocked)
    NOTE: this consensus-count path has no per-book reads, so the full dispersion/scatter
    gate (see market_quality) can't run here — only the count-downgrade is removed. If this
    legacy path is revived for live use, plumb per-book dispersion through for the scatter gate.
    """
    if mp.mapping_status != "mapped_exact":
        return mp
    n = mp.all_num_books if mp.all_num_books is not None else 0
    mp.liquidity_flag = "ok" if n >= thin_books_threshold else "thin"   # count no longer downgrades
    return mp


def _map_team_win(target_team: str | None, consensus: pd.DataFrame) -> Mapping:
    if not target_team:
        return Mapping(
            mapped_market="h2h",
            mapped_outcome=None,
            mapped_line=None,
            mapped_bet_description="h2h / unspecified team",
            market_prob=None,
            mapping_status="ambiguous_review",
            liquidity_flag="n/a",
        )

    h2h = _slice_market(consensus, "h2h", line=None)
    if h2h.empty:
        return _no_market("h2h", target_team)

    rows = h2h[h2h["outcome"].str.lower() == target_team.lower()]
    if rows.empty:
        return Mapping(
            mapped_market="h2h",
            mapped_outcome=target_team,
            mapped_line=None,
            mapped_bet_description=(
                f"h2h / {target_team} (target_team did not match any "
                "consensus outcome — check spelling vs API names)"
            ),
            market_prob=None,
            mapping_status="ambiguous_review",
            liquidity_flag="n/a",
        )

    return _row_to_mapping(
        rows.iloc[0],
        market_key="h2h",
        outcome=target_team,
        line=None,
        description=f"h2h / {target_team} / full-time result",
    )


def _map_match_total_over(line: float | None, consensus: pd.DataFrame) -> Mapping:
    # "Will the match have 3 or more goals?" ↔ Over 2.5
    eff_line = 2.5 if line is None else line
    totals = _slice_market(consensus, "totals", line=eff_line)
    over = totals[totals["outcome"].str.lower() == "over"]
    if over.empty:
        return _no_market("totals", f"Over {eff_line}", line=eff_line)
    return _row_to_mapping(
        over.iloc[0],
        market_key="totals",
        outcome="Over",
        line=eff_line,
        description=f"totals / Over {eff_line} goals",
    )


def _map_match_total_under(line: float | None, consensus: pd.DataFrame) -> Mapping:
    # "Will the match have 2 or fewer goals?" ↔ Under 2.5
    eff_line = 2.5 if line is None else line
    totals = _slice_market(consensus, "totals", line=eff_line)
    under = totals[totals["outcome"].str.lower() == "under"]
    if under.empty:
        return _no_market("totals", f"Under {eff_line}", line=eff_line)
    return _row_to_mapping(
        under.iloc[0],
        market_key="totals",
        outcome="Under",
        line=eff_line,
        description=f"totals / Under {eff_line} goals",
    )


def _map_btts(consensus: pd.DataFrame) -> Mapping:
    btts = _slice_market(consensus, "btts", line=None)
    yes = btts[btts["outcome"].str.lower() == "yes"]
    if yes.empty:
        return _no_market("btts", "Yes")
    return _row_to_mapping(
        yes.iloc[0],
        market_key="btts",
        outcome="Yes",
        line=None,
        description="btts / Yes (both teams score)",
    )


def _map_halftime_draw(consensus: pd.DataFrame) -> Mapping:
    for mkey in HALFTIME_H2H_KEYS:
        ht = _slice_market(consensus, mkey, line=None)
        draw = ht[ht["outcome"].str.lower() == "draw"]
        if not draw.empty:
            return _row_to_mapping(
                draw.iloc[0],
                market_key=mkey,
                outcome="Draw",
                line=None,
                description=f"{mkey} / Draw (tied at halftime)",
            )
    return _no_market(HALFTIME_H2H_KEYS[0], "Draw")


def map_question(
    question_type: str | None,
    target_team: str | None,
    target_player: str | None,
    line: float | None,
    consensus: pd.DataFrame,
    min_books: int = 3,
    thin_books_threshold: int = 5,
) -> Mapping:
    """Return a Mapping for one SportsPredict question.

    ``consensus`` should already be filtered to the rows for that match.

    ``min_books`` / ``thin_books_threshold`` gate per-event-friendly
    liquidity checks (see ``_apply_liquidity_gate``). They apply to all
    market types uniformly — bulk markets like h2h typically have 15-20
    books and easily clear ``thin_books_threshold``; per-event markets like
    btts may be thinner and are where this gate actually bites.
    """
    qt = (question_type or "").strip().lower()

    if qt == "team_win":
        mp = _map_team_win(target_team, consensus)
    elif qt == "match_total_over":
        mp = _map_match_total_over(line, consensus)
    elif qt == "match_total_under":
        mp = _map_match_total_under(line, consensus)
    elif qt == "both_teams_score":
        mp = _map_btts(consensus)
    elif qt == "halftime_draw":
        mp = _map_halftime_draw(consensus)
    elif qt in NEEDS_MODEL:
        mp = _empty("needs_model", line=line)
    elif qt in UNMAPPED_SKIP:
        mp = _empty("unmapped_skip", line=line)
    else:
        mp = _empty("ambiguous_review", line=line)

    return _apply_liquidity_gate(mp, min_books, thin_books_threshold)
