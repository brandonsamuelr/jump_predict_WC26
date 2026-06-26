"""Guards for total_cards_over direct-market pricing (card-count convention).

    .venv/bin/python tests/test_cards.py

Asserts: total_cards_over routes to the cards-market path and submits the
de-vigged market price UNDISTORTED; comparison/period card rows stay shadow;
missing market -> safe shadow fallback; "N or more" maps to Over (N-0.5).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib import slate, cards_pricing as CD
from odds_lib.optimizer import optimize
from odds_lib.edge import classify, K_PRIOR


def _book(title, line, over_px, under_px):
    return {"title": title, "markets": [{"key": CD.CARDS_TOTAL_MARKET, "outcomes": [
        {"name": "Over", "point": line, "price": over_px},
        {"name": "Under", "point": line, "price": under_px}]}]}


def _cards_game(line=3.5, n_books=6):
    return {"bookmakers": [_book(f"B{i}", line, -120, 100) for i in range(n_books)]}


# --- 1. total_cards_over prices off market, undistorted ----------------------

def test_total_cards_over_prices_off_market_undistorted():
    g = _cards_game(line=3.5, n_books=6)
    row = {"question_type": "total_cards_over", "target_team": "", "line": "3.5"}
    tier, p, mp = slate.resolve_row(row, None, g, None)
    assert tier == "CARDS_OK" and classify(tier, "total_cards_over") == ("CARDS", "ok")
    cd = CD.price_cards_over(g, 3.5)
    assert abs(p - cd.p_over) < 1e-9 and 0.50 < p < 0.60   # ~0.545 de-vigged
    s = optimize(tier=tier, question_type="total_cards_over", p_hat=p, shadow=0.49)
    assert abs(s.q - p) < 1e-9   # market price UNDISTORTED (k=1), not pulled to 0.49

def test_few_agreeing_cards_books_used_undistorted():
    # 4 books that AGREE (identical odds -> disp 0) -> 'ok' (confident), USED, value=market.
    # Flag is now AGREEMENT-based not count-based; both ok/thin are k=1 (no distortion).
    g = _cards_game(line=3.5, n_books=4)
    row = {"question_type": "total_cards_over", "target_team": "", "line": "3.5"}
    tier, p, _ = slate.resolve_row(row, None, g, None)
    assert tier in ("CARDS_OK", "CARDS_THIN") and K_PRIOR[("CARDS", "ok")] == 1.00
    s = optimize(tier=tier, question_type="total_cards_over", p_hat=p, shadow=0.49)
    assert abs(s.q - CD.price_cards_over(g, 3.5).p_over) < 1e-9   # market price UNDISTORTED


# --- 2. comparison / period card rows STAY shadow ---------------------------

def test_more_cards_without_consensus_stays_shadow():
    # team_more_cards needs the de-vigged consensus (favorite_gap); c=None -> shadow.
    g = _cards_game()
    tier, p, _ = slate.resolve_row({"question_type": "team_more_cards", "target_team": "Croatia", "line": ""},
                                   None, g, None)
    assert tier == "PENDING" and p is None

def test_2h_cards_market_derived_when_card_market_present():
    # The proper controlling variable: full-match cards market lambda x 2H share (NOT favorite_gap,
    # NOT the club-only floor). With a cards market present -> CARDS_2H_MKT, per-match.
    g = _cards_game()   # alternate_totals_cards present
    for qt, ln in (("team_card_2h", ""), ("team_cards_2h_over", "1.5"), ("total_cards_2h_over", "3.5")):
        tier, p, _ = slate.resolve_row({"question_type": qt, "target_team": "Croatia", "line": ln}, None, g, None)
        assert tier == "CARDS_2H_MKT" and p is not None and 0.02 < p < 0.98, f"{qt}: {tier} {p}"
    assert classify("CARDS_2H_MKT", "x") == ("CARDS_2H", "market") and K_PRIOR[("CARDS_2H", "market")] == 1.0
    # total > team at same line (team gets the 0.5 split) and 2H value moves with the market lambda
    _, p_tot, _ = slate.resolve_row({"question_type": "total_cards_2h_over", "target_team": "", "line": "2.5"}, None, g, None)
    _, p_team, _ = slate.resolve_row({"question_type": "team_cards_2h_over", "target_team": "X", "line": "2.5"}, None, g, None)
    assert p_tot > p_team

def test_2h_cards_no_market_falls_back_to_club_floor():
    # no cards market -> last-resort CLUB-only floor (flagged mis-populated), never shadow/0.50
    g = {"bookmakers": []}
    tier, p, _ = slate.resolve_row({"question_type": "total_cards_2h_over", "target_team": "", "line": "3.5"}, None, g, None)
    assert tier == "CARDS_2H_FLOOR" and p is not None and p != 0.50


# --- 3. safe fallback -------------------------------------------------------

def test_missing_cards_market_falls_back_to_shadow():
    row = {"question_type": "total_cards_over", "target_team": "", "line": "3.5"}
    tier, p, _ = slate.resolve_row(row, None, {"bookmakers": []}, None)
    assert tier == "PENDING" and p is None

def test_line_not_quoted_falls_back_to_shadow():
    g = _cards_game(line=4.5, n_books=6)   # market only at 4.5
    row = {"question_type": "total_cards_over", "target_team": "", "line": "3.5"}  # need 3.5
    tier, p, _ = slate.resolve_row(row, None, g, None)
    assert tier == "PENDING" and p is None   # no 3.5 quote -> shadow, no approximation


# --- 4. line mapping --------------------------------------------------------

def test_line_mapping_n_or_more_is_over_n_minus_half():
    # "4 or more cards" -> line 3.5; must read the 3.5 line, not 4.5
    g = {"bookmakers": [_book("B0", 3.5, -120, 100), _book("B1", 4.5, 130, -150)]}
    cd = CD.price_cards_over(g, 3.5)
    assert cd.mapped and cd.n_books == 1 and abs(cd.line - 3.5) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
