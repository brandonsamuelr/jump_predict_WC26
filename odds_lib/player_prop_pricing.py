"""Market-derived probabilities from ONE-SIDED player-prop markets.

Distinct from :mod:`player_features` (features only, never p_truth): this
module DOES produce a market-implied probability — but only from a market
that measures the *same event* as the contest question, and always flagged
as an approximation, never as a clean two-sided de-vig.

STRICT market-equivalence (the whole point of this module)
----------------------------------------------------------
A contest player-prop ``question_type`` may only be priced from the API
market that measures the identical event:

  - ``player_goal``      <- ``player_goal_scorer_anytime``        (direct)
  - ``player_sot_over``  <- ``player_shots_on_target`` @ point 0.5 (direct)
  - ``player_goal_or_assist`` <- ``player_goal_scorer_anytime``
        (PARTIAL: anytime-scorer is a *lower bound* on goal-or-assist —
         a scorer always qualifies, an assister-only does not. Returned
         with confidence="partial" and never as clean p_truth.)
  - ``player_sot_2h_over`` <- NO EQUIVALENT MARKET in the soccer feed
        (full-match SOT != 2H SOT; anytime-scorer != SOT). Returns
        UNSUPPORTED. May be used as a weak *feature* elsewhere, never here
        as p.

Anything else, or a player the book did not quote, returns UNSUPPORTED with
a reason. We never fabricate a missing No-side and never cross-map prop
types.

One-sided vig
-------------
These markets are quoted Yes-only (e.g. anytime-scorer gives a price for
"Yes", no "No"). The implied probability is therefore vig-inclusive and
cannot be de-vigged exactly without the other side. We report:

  - ``market_prob_raw``          : consensus vig-inclusive implied prob
                                   (sharp-book subset preferred when present)
  - ``market_prob_vig_adjusted`` : ``raw / overround`` using a documented,
                                   market-typical single-selection overround

and tag the source ``one_sided_prop_market_adjusted`` so downstream code can
never mistake it for a clean two-sided market price.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, asdict
from typing import Any

from .odds import odds_to_prob, DEFAULT_SHARP_BOOKS


# --- Documented one-sided overround haircuts -------------------------------
# A Yes-only quote's implied prob carries vig we cannot strip exactly. We
# divide by a documented, conservative single-selection overround. These are
# market-typical assumptions, NOT derived from the data, and the output is
# flagged accordingly. Override per call if you have a better estimate.
ANYTIME_SCORER_OVERROUND = 1.12  # scorer markets carry heavy per-selection vig
SOT_OVERROUND = 1.06             # binary over/under 0.5 SOT, tighter book margin

SOURCE_TAG = "one_sided_prop_market_adjusted"


@dataclass(frozen=True)
class PropSpec:
    api_market: str
    needs_point: bool
    point: float | None
    overround: float
    confidence: str  # "direct" | "partial"
    note: str = ""
    # True ONLY for mappings VERIFIED to be a directional lower bound whose
    # market definition/timeframe matches the contest question (so the bound
    # P(question) >= P(market) holds by event structure). Set per-mapping after
    # verification — never inferred from confidence alone. Enables the
    # definitional lower-bound clamp in optimizer.optimize().
    lower_bound: bool = False


# The ONLY allowed contest-question -> API-market equivalences. A value of
# None means "no equivalent market exists" -> always UNSUPPORTED.
PROP_EQUIVALENCE: dict[str, PropSpec | None] = {
    "player_goal": PropSpec(
        api_market="player_goal_scorer_anytime",
        needs_point=False,
        point=None,
        overround=ANYTIME_SCORER_OVERROUND,
        confidence="direct",
    ),
    "player_sot_over": PropSpec(
        api_market="player_shots_on_target",
        needs_point=True,
        point=0.5,
        overround=SOT_OVERROUND,
        confidence="direct",
        note="1+ SOT == Over 0.5 shots on target",
    ),
    "player_goal_or_assist": PropSpec(
        api_market="player_goal_scorer_anytime",
        needs_point=False,
        point=None,
        overround=ANYTIME_SCORER_OVERROUND,
        confidence="partial",
        note="anytime-scorer is a LOWER BOUND on goal-or-assist; partial only",
        # VERIFIED lower bound: scoring-or-assisting is a SUPERSET of scoring,
        # and both the contest question and the anytime-goal market are
        # full-match, excluding own goals -> same timeframe/definition. So
        # P(goal or assist) >= P(anytime goal) by event structure, at any
        # sample size. (Would NOT hold if the question were half-specific vs a
        # full-match market — do not set lower_bound for such a mapping.)
        lower_bound=True,
    ),
    # Full-match SOT is NOT 2H SOT; anytime-scorer is NOT SOT. No equivalent
    # market in the soccer feed -> never priced here.
    "player_sot_2h_over": None,
}


@dataclass
class PropPricing:
    question_type: str
    target_player: str | None
    line: float | None
    # resolution
    mapped: bool
    status: str  # mapped_one_sided | partial_one_sided | unsupported_*
    confidence: str  # direct | partial | none
    api_market: str | None
    source_tag: str | None
    # probabilities
    market_prob_raw: float | None
    market_prob_vig_adjusted: float | None
    overround_used: float | None
    # liquidity
    book_count: int
    sharp_book_count: int
    books_used: str
    liquidity_flag: str  # ok | thin | low | n/a
    note: str = ""
    lower_bound: bool = False  # market_prob is a definitional lower bound on the question

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_lower_bound_prop(question_type: str) -> bool:
    """True iff the question_type maps to a market that is a VERIFIED
    definitional lower bound (e.g. goal-or-assist priced off anytime-goal).
    Used by the submission path to enforce SUBMIT >= p_hat for these rows."""
    spec = PROP_EQUIVALENCE.get((question_type or "").strip().lower())
    return bool(spec is not None and spec.lower_bound)


# --- name matching ---------------------------------------------------------

_CONNECTORS = {"al", "el", "de", "da", "di", "van", "von", "the"}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _token_list(name: str) -> list[str]:
    """Ordered tokens of a name (accent/case-stripped). Order-deterministic
    so any positional logic (e.g. surname) cannot vary across processes."""
    norm = _strip_accents(name or "").lower()
    return "".join(c if c.isalpha() else " " for c in norm).split()


def match_player(query: str, candidates: list[str]) -> str | None:
    """Match a query player name to a book outcome description — PRECISION
    over recall, by design.

    Requires EVERY distinctive query token (len>=3, non-connector) to appear
    in the candidate's tokens. So "Mohanad Ali" matches "Mohanad Ali" but
    never "Hussein Ali"; "Moussa Al Taamari" never matches "Anis Hadj Moussa"
    or the differently-spelled "Musa Al Tamari". When nothing fully matches we
    return None (caller marks it unsupported) rather than guess via a surname
    — common surname tokens ("Ali", "Moussa") make that unsafe, and a wrong
    player is far worse than a skip.
    """
    qtoks = {t for t in _token_list(query) if len(t) >= 3 and t not in _CONNECTORS}
    if not qtoks:
        return None
    cand_tokens = {c: set(_token_list(c)) for c in candidates}
    full = [c for c, toks in cand_tokens.items() if qtoks <= toks]
    if not full:
        return None
    # deterministic: most token overlap, then shortest name, then name order
    return min(full, key=lambda c: (-len(qtoks & cand_tokens[c]), len(c), c))


# --- quote extraction ------------------------------------------------------


@dataclass
class _Quote:
    book: str
    implied_raw: float
    is_sharp: bool


def _extract_quotes(
    game: dict,
    api_market: str,
    player: str,
    point: float | None,
    sharp_books: tuple[str, ...],
) -> tuple[list[_Quote], list[str]]:
    """Pull Yes/Over-side quotes for one player+market+point across books.

    Returns ``(quotes, all_candidate_player_names)``. De-dupes books that
    appear under several region keys (keeps the first). Never invents a
    No-side.
    """
    # First, gather candidate player names present in this market so the
    # caller can report "player not quoted" vs "market absent" distinctly.
    candidates: set[str] = set()
    by_book: dict[str, _Quote] = {}

    for bk in game.get("bookmakers", []):
        title = bk.get("title", "")
        for mkt in bk.get("markets", []):
            if mkt.get("key") != api_market:
                continue
            # collect candidates and find this player's outcome
            outcomes = mkt.get("outcomes", [])
            local_names = [
                (o.get("description") or o.get("name") or "") for o in outcomes
            ]
            candidates.update(n for n in local_names if n)
            matched_name = match_player(player, local_names)
            if matched_name is None:
                continue
            for o in outcomes:
                desc = o.get("description") or o.get("name") or ""
                if desc != matched_name:
                    continue
                side = (o.get("name") or "").strip().lower()
                # Yes-only: keep the affirmative side; skip explicit No/Under.
                if side in ("no", "under"):
                    continue
                if point is not None:
                    pt = o.get("point")
                    if pt is None or float(pt) != float(point):
                        continue
                price = int(o["price"])
                implied = float(odds_to_prob([price])[0])
                # de-dupe by book title (region duplicates)
                if title not in by_book:
                    by_book[title] = _Quote(
                        book=title,
                        implied_raw=implied,
                        is_sharp=title in sharp_books,
                    )
    return list(by_book.values()), sorted(candidates)


def _liquidity_flag(book_count: int, sharp_count: int) -> str:
    if book_count == 0:
        return "n/a"
    if sharp_count >= 1 and book_count >= 3:
        return "ok"
    if book_count >= 1:
        return "thin"
    return "low"


def price_player_prop(
    question_type: str,
    target_player: str | None,
    line: float | None,
    game: dict,
    sharp_books: tuple[str, ...] = DEFAULT_SHARP_BOOKS,
) -> PropPricing:
    """Price one contest player-prop question from a single game's odds JSON.

    Enforces strict market-equivalence (see module docstring). Returns a
    :class:`PropPricing`; ``mapped`` is False for every UNSUPPORTED reason.
    """
    qt = (question_type or "").strip().lower()

    def _unsupported(status: str, note: str) -> PropPricing:
        return PropPricing(
            question_type=qt, target_player=target_player, line=line,
            mapped=False, status=status, confidence="none",
            api_market=None, source_tag=None,
            market_prob_raw=None, market_prob_vig_adjusted=None,
            overround_used=None, book_count=0, sharp_book_count=0,
            books_used="", liquidity_flag="n/a", note=note,
        )

    if qt not in PROP_EQUIVALENCE:
        return _unsupported("unsupported_unknown_prop_type",
                            f"{qt!r} is not a known player-prop question type")
    spec = PROP_EQUIVALENCE[qt]
    if spec is None:
        return _unsupported(
            "unsupported_no_equivalent_market",
            f"{qt!r} has no equivalent two-sided market in the feed "
            "(full-match SOT and anytime-scorer are NOT 2H-SOT); feature-only",
        )
    if not target_player:
        return _unsupported("unsupported_no_player", "no target_player given")

    quotes, candidates = _extract_quotes(
        game, spec.api_market, target_player, spec.point, sharp_books
    )
    if not candidates:
        return _unsupported(
            "unsupported_market_absent",
            f"book feed has no {spec.api_market} market for this game",
        )
    if not quotes:
        return _unsupported(
            "unsupported_player_not_quoted",
            f"{target_player!r} not quoted in {spec.api_market} "
            f"(market has {len(candidates)} other players)",
        )

    sharp_quotes = [q for q in quotes if q.is_sharp]
    chosen = sharp_quotes if sharp_quotes else quotes
    raw = sum(q.implied_raw for q in chosen) / len(chosen)
    adj = raw / spec.overround

    status = "mapped_one_sided" if spec.confidence == "direct" else "partial_one_sided"
    return PropPricing(
        question_type=qt, target_player=target_player, line=line,
        mapped=True, status=status, confidence=spec.confidence,
        api_market=spec.api_market, source_tag=SOURCE_TAG,
        market_prob_raw=round(raw, 4),
        market_prob_vig_adjusted=round(adj, 4),
        overround_used=spec.overround,
        book_count=len(quotes),
        sharp_book_count=len(sharp_quotes),
        books_used=", ".join(sorted(q.book for q in quotes)),
        liquidity_flag=_liquidity_flag(len(quotes), len(sharp_quotes)),
        note=spec.note,
        lower_bound=spec.lower_bound,
    )


__all__ = [
    "PropSpec",
    "PropPricing",
    "PROP_EQUIVALENCE",
    "ANYTIME_SCORER_OVERROUND",
    "SOT_OVERROUND",
    "SOURCE_TAG",
    "match_player",
    "price_player_prop",
    "is_lower_bound_prop",
]
