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


# --- TIERED prop de-vig (measure, don't type) ------------------------------
# A prop's implied prob is vig-inclusive. We strip the margin by the BEST
# available source, in priority order; the per-prop choice is recorded in
# PropPricing.overround_source so it is auditable, never silent:
#   a. exact_two_sided        -- the SAME player has BOTH sides quoted: de-vig
#                                exactly per book (p = aff/(aff+neg)). No prior.
#   b. same_slate_market_prior-- target one-sided, but OTHER players in the same
#                                game+market have both sides: use their median
#                                two-sided booksum as a market-specific overround.
#                                (Selection-biased toward liquid/favorite players
#                                -> overround_source makes it auditable vs tier a.)
#   c. global_player_prop_prior- nothing two-sided anywhere: the MEASURED global
#                                player-prop booksum prior below.
# NO privileged per-market constants (the old 1.06/1.10/1.12). The global prior
# is MEASURED from our own feed (median two-sided booksum across 177 two-sided
# anytime-scorer quotes = 1.045), refreshable -- not a typed guess.
GLOBAL_PLAYER_PROP_OVERROUND = 1.045

# R32 SHADOW ONLY (NOT a tuning target): for each resolved prop we log what each
# flat candidate would have produced next to the tiered output + outcome, to test
# whether the TIERED logic beats any flat constant -- never to refit a constant on
# a small sample (the n=9 SOT trap). See odds_lib/prop_devig_shadow.py.
SHADOW_OVERROUND_CANDIDATES = (1.045, 1.06, 1.10, 1.12)

SOURCE_TAG = "prop_market_tiered_devig"


@dataclass(frozen=True)
class PropSpec:
    api_market: str
    needs_point: bool
    point: float | None
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
        confidence="direct",
    ),
    "player_sot_over": PropSpec(
        api_market="player_shots_on_target",
        needs_point=True,
        point=0.5,
        confidence="direct",
        note="1+ SOT == Over 0.5 shots on target",
    ),
    "player_goal_or_assist": PropSpec(
        api_market="player_goal_scorer_anytime",
        needs_point=False,
        point=None,
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
    source: str = ""           # "" | "direct" | "proxy_floor" (goal_or_assist routing)
    floor_prob: float | None = None  # anytime-goal lower-bound value (for audit)
    # de-vig provenance (step 5): which tier produced overround_used, and how many
    # samples backed it. exact_two_sided | same_slate_market_prior | global_player_prop_prior.
    # prior_n = both-sided books (tier a) or same-slate booksum samples (tier b).
    overround_source: str = ""
    overround_prior_n: int = 0

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
class _Sided:
    book: str
    aff: float            # affirmative (Over/Yes) implied prob
    neg: float | None     # negative (Under/No) implied prob, None if one-sided
    is_sharp: bool


def _extract_sided(game, api_market, player, point, sharp_books):
    """Per-book (aff, neg) implied for the target player, candidate names, and
    the same-slate/market two-sided booksums from OTHER players (the tier-b prior).

    aff = Over/Yes side; neg = Under/No side (None if the book is one-sided).
    The No side is CAPTURED now (the old extractor discarded it) -- that is what
    makes exact two-sided de-vig possible."""
    candidates: set[str] = set()
    by_book: dict[str, _Sided] = {}
    slate_booksums: list[float] = []

    for bk in game.get("bookmakers", []):
        title = bk.get("title", "")
        for mkt in bk.get("markets", []):
            if mkt.get("key") != api_market:
                continue
            permap: dict[str, dict[str, float]] = {}
            for o in mkt.get("outcomes", []):
                desc = o.get("description") or o.get("name") or ""
                if not desc:
                    continue
                candidates.add(desc)
                if point is not None:
                    pt = o.get("point")
                    if pt is None or float(pt) != float(point):
                        continue
                try:
                    ip = float(odds_to_prob([int(o["price"])])[0])
                except (KeyError, ValueError, TypeError):
                    continue
                side = (o.get("name") or "").strip().lower()
                d = permap.setdefault(desc, {})
                if side in ("over", "yes"):
                    d["aff"] = ip
                elif side in ("under", "no"):
                    d["neg"] = ip
            matched = match_player(player, list(permap.keys()))
            # tier-b prior samples: OTHER players quoted two-sided in this book
            for nm, d in permap.items():
                if nm != matched and "aff" in d and "neg" in d and (d["aff"] + d["neg"]) > 0:
                    slate_booksums.append(d["aff"] + d["neg"])
            if matched and "aff" in permap.get(matched, {}) and title not in by_book:
                by_book[title] = _Sided(book=title, aff=permap[matched]["aff"],
                                        neg=permap[matched].get("neg"),
                                        is_sharp=title in sharp_books)
    return by_book, sorted(candidates), slate_booksums


def _liquidity_flag(book_count: int, sharp_count: int) -> str:
    if book_count == 0:
        return "n/a"
    if sharp_count >= 1 and book_count >= 3:
        return "ok"
    if book_count >= 1:
        return "thin"
    return "low"


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def devig_tiered(game, api_market, target_player, point, sharp_books):
    """TIERED prop de-vig. Returns ``(result | None, status)``.

    status: "ok" | "market_absent" | "player_not_quoted". On "ok", result carries
    raw, adj (de-vigged p), overround, source, prior_n, book_count, sharp, books,
    liquidity. Priority: exact_two_sided > same_slate_market_prior > global prior.
    NO per-market privileged constant -- a SOT prop and a scorer prop with the same
    de-vig context get the SAME treatment."""
    by_book, candidates, slate_booksums = _extract_sided(
        game, api_market, target_player, point, sharp_books)
    if not candidates:
        return None, "market_absent"
    if not by_book:
        return None, "player_not_quoted"
    sharp = [s for s in by_book.values() if s.is_sharp]
    chosen = sharp if sharp else list(by_book.values())
    raw = sum(s.aff for s in chosen) / len(chosen)
    both = [s for s in chosen if s.neg is not None]
    if both:                                          # tier a: exact two-sided
        adj = sum(s.aff / (s.aff + s.neg) for s in both) / len(both)
        overround = sum(s.aff + s.neg for s in both) / len(both)
        source, prior_n = "exact_two_sided", len(both)
    elif slate_booksums:                              # tier b: same-slate prior
        overround = _median(slate_booksums)
        adj = raw / overround
        source, prior_n = "same_slate_market_prior", len(slate_booksums)
    else:                                             # tier c: global measured prior
        overround = GLOBAL_PLAYER_PROP_OVERROUND
        adj = raw / overround
        source, prior_n = "global_player_prop_prior", 0
    bc, sc = len(by_book), len(sharp)
    return {
        "raw": raw, "adj": adj, "overround": overround,
        "source": source, "prior_n": prior_n,
        "book_count": bc, "sharp": sc,
        "books": ", ".join(sorted(by_book.keys())),
        "liquidity": _liquidity_flag(bc, sc),
    }, "ok"


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

    # goal-or-assist: prefer the DIRECT score-or-assist market when present, with
    # the anytime-goal value as a definitional lower-bound guard; fall back to the
    # anytime-goal LOWER BOUND proxy (+ submission clamp) when no direct market.
    # Both legs use the TIERED de-vig -- anytime-scorer routes to exact two-sided
    # where books quote the No side, which is what corrects the old 1.12 over-strip.
    if qt == "player_goal_or_assist":
        direct, _ = devig_tiered(game, "player_to_score_or_assist",
                                 target_player, None, sharp_books)
        floor, _ = devig_tiered(game, "player_goal_scorer_anytime",
                                target_player, None, sharp_books)
        if direct is None and floor is None:
            return _unsupported("unsupported_market_absent",
                                "no score_or_assist and no anytime-goal market for this player/game")
        if direct is not None:
            floor_adj = floor["adj"] if floor else None
            p = max(direct["adj"], floor_adj) if floor_adj is not None else direct["adj"]
            floor_active = floor_adj is not None and floor_adj >= direct["adj"]
            return PropPricing(
                question_type=qt, target_player=target_player, line=line,
                mapped=True, status="goal_or_assist_direct", confidence="direct",
                api_market="player_to_score_or_assist", source_tag=SOURCE_TAG,
                market_prob_raw=round(direct["raw"], 4), market_prob_vig_adjusted=round(p, 4),
                overround_used=round(direct["overround"], 4),
                book_count=direct["book_count"], sharp_book_count=direct["sharp"],
                books_used=direct["books"], liquidity_flag=direct["liquidity"],
                note=(f"DIRECT score_or_assist; anytime_floor="
                      f"{None if floor_adj is None else round(floor_adj, 4)}; floor_active={floor_active}"),
                lower_bound=False, source="direct",
                floor_prob=(None if floor_adj is None else round(floor_adj, 4)),
                overround_source=direct["source"], overround_prior_n=direct["prior_n"],
            )
        return PropPricing(   # floor-only fallback: anytime-goal LOWER BOUND proxy
            question_type=qt, target_player=target_player, line=line,
            mapped=True, status="goal_or_assist_proxy_floor", confidence="partial",
            api_market="player_goal_scorer_anytime", source_tag=SOURCE_TAG,
            market_prob_raw=round(floor["raw"], 4), market_prob_vig_adjusted=round(floor["adj"], 4),
            overround_used=round(floor["overround"], 4),
            book_count=floor["book_count"], sharp_book_count=floor["sharp"],
            books_used=floor["books"], liquidity_flag=floor["liquidity"],
            note="no direct score_or_assist market; anytime-goal LOWER BOUND proxy (+ clamp)",
            lower_bound=True, source="proxy_floor", floor_prob=round(floor["adj"], 4),
            overround_source=floor["source"], overround_prior_n=floor["prior_n"],
        )

    res, status = devig_tiered(game, spec.api_market, target_player, spec.point, sharp_books)
    if status == "market_absent":
        return _unsupported(
            "unsupported_market_absent",
            f"book feed has no {spec.api_market} market for this game",
        )
    if status == "player_not_quoted":
        return _unsupported(
            "unsupported_player_not_quoted",
            f"{target_player!r} not quoted in {spec.api_market}",
        )

    # status_tag records the de-vig quality: a two-sided exact de-vig is a CLEANER
    # read than a one-sided prior-adjusted one, even for a "direct" mapping.
    if res["source"] == "exact_two_sided":
        status_tag = "mapped_two_sided"
    else:
        status_tag = "mapped_one_sided" if spec.confidence == "direct" else "partial_one_sided"
    return PropPricing(
        question_type=qt, target_player=target_player, line=line,
        mapped=True, status=status_tag, confidence=spec.confidence,
        api_market=spec.api_market, source_tag=SOURCE_TAG,
        market_prob_raw=round(res["raw"], 4),
        market_prob_vig_adjusted=round(res["adj"], 4),
        overround_used=round(res["overround"], 4),
        book_count=res["book_count"],
        sharp_book_count=res["sharp"],
        books_used=res["books"],
        liquidity_flag=res["liquidity"],
        note=spec.note,
        lower_bound=spec.lower_bound,
        overround_source=res["source"], overround_prior_n=res["prior_n"],
    )


# Benched-player prop: a confirmed-bench player's true-P is NOT the starter read (assumes 90'),
# nor a shadow constant. It's a principled minutes-scaled closed form:
#   P(event) = P(appears) * [1 - (1 - p_starter)^(sub_minutes / 90)]
# where p_starter is the de-vigged STARTER market read (the book prices him as if he plays a full
# match). (appearance_prob, expected_sub_minutes) keyed on bench-usage status. out_of_squad -> None
# (no appearance -> route stays PENDING). Defaults: a rested star (high usage) ~0.85 appearance/35',
# scaling down for lower-usage / unknown. Documented, transfer-safe (no team identity).
SUB_PROFILE: dict[str, tuple[float, float]] = {
    "bench_high_usage": (0.85, 35.0),
    "bench_unknown":    (0.70, 28.0),
    "bench_low_usage":  (0.45, 20.0),
}


def minutes_scaled_sub(p_starter: float | None, status: str) -> float | None:
    """Minutes-scaled benched-player prob; None if status is not a sub-eligible bench status."""
    if p_starter is None:
        return None
    prof = SUB_PROFILE.get((status or "").strip().lower())
    if prof is None:
        return None
    appearance, minutes = prof
    return float(appearance * (1.0 - (1.0 - float(p_starter)) ** (minutes / 90.0)))


__all__ = [
    "PropSpec",
    "PropPricing",
    "SUB_PROFILE",
    "minutes_scaled_sub",
    "PROP_EQUIVALENCE",
    "GLOBAL_PLAYER_PROP_OVERROUND",
    "SHADOW_OVERROUND_CANDIDATES",
    "SOURCE_TAG",
    "match_player",
    "devig_tiered",
    "price_player_prop",
    "is_lower_bound_prop",
]
