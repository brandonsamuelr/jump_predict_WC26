"""Guards for CHANGE 3 — corners direct-market pricing (count rows only).

    .venv/bin/python tests/test_corners.py

Asserts: count rows (team_corners_over/total_corners_over) route to a market-priced
CORNERS tier and land NEAR the de-vigged market (not the placeholder shadow);
comparison/period corners STAY shadow; missing market -> safe shadow fallback;
"N or more" maps to the Over (N-0.5) line.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib import slate, corners_pricing as CN
from odds_lib.optimizer import optimize
from odds_lib.edge import classify, K_PRIOR


def _book(title, market_key, line, over_px, under_px, team=None):
    oc = [{"name": "Over", "point": line, "price": over_px},
          {"name": "Under", "point": line, "price": under_px}]
    if team is not None:
        for o in oc:
            o["description"] = team
    return {"title": title, "markets": [{"key": market_key, "outcomes": oc}]}


def _team_corners_game(team="Ghana", line=4.5, n_books=5):
    # n_books quoting team corners Over/Under at `line` (de-vig ~0.545 at -120/+100)
    bms = [_book(f"B{i}", CN.TEAM_CORNERS_MARKET, line, -120, 100, team=team)
           for i in range(n_books)]
    return {"bookmakers": bms}


def _total_corners_game(line=8.5, n_books=6):
    bms = [_book(f"B{i}", CN.TOTAL_CORNERS_MARKET, line, -110, -110) for i in range(n_books)]
    return {"bookmakers": bms}


# --- 1. count rows route to market-priced tier, land near the price ---------

def test_team_corners_over_prices_off_market():
    g = _team_corners_game(team="Ghana", line=4.5, n_books=5)
    row = {"question_type": "team_corners_over", "target_team": "Ghana", "line": "4.5"}
    tier, p, mp = slate.resolve_row(row, None, g, None)
    assert tier == "CORNERS_OK" and classify(tier, "team_corners_over") == ("CORNERS", "ok")
    cp = CN.price_corners_over(g, CN.TEAM_CORNERS_MARKET, "Ghana", 4.5)
    assert abs(p - cp.p_over) < 1e-9 and 0.50 < p < 0.60   # ~0.545
    s = optimize(tier=tier, question_type="team_corners_over", p_hat=p, shadow=0.392)
    assert abs(s.q - p) < 1e-9   # submits the market price UNDISTORTED (k=1), not pulled to 0.392

def test_total_corners_over_prices_off_market():
    g = _total_corners_game(line=8.5, n_books=6)
    row = {"question_type": "total_corners_over", "target_team": "", "line": "8.5"}
    tier, p, _ = slate.resolve_row(row, None, g, None)
    assert tier == "CORNERS_OK" and 0.0 < p < 1.0


def test_corners_flag_is_diagnostic_only_value_is_market_price():
    # 4 books that AGREE -> 'ok' (confident; flag is agreement-based now, not count). Both
    # CORNERS ok/thin are k=1 -> the flag is diagnostic and never distorts the value.
    g = _team_corners_game(team="Ghana", line=4.5, n_books=4)
    row = {"question_type": "team_corners_over", "target_team": "Ghana", "line": "4.5"}
    tier, p, _ = slate.resolve_row(row, None, g, None)
    assert tier in ("CORNERS_OK", "CORNERS_THIN")
    assert K_PRIOR[("CORNERS", "ok")] == 1.00 and K_PRIOR[("CORNERS", "thin")] == 1.00
    cp = CN.price_corners_over(g, CN.TEAM_CORNERS_MARKET, "Ghana", 4.5)
    s = optimize(tier=tier, question_type="team_corners_over", p_hat=p, shadow=0.392)
    assert abs(s.q - cp.p_over) < 1e-9                  # submits market price undistorted


# --- 2. comparison / period corners STAY shadow -----------------------------

def test_comparison_corners_without_inputs_stay_shadow():
    g = _team_corners_game()  # team_corners market present, but no corners_1x2; c=None
    for qt in ("team_more_corners_full", "second_half_corners_over"):
        row = {"question_type": qt, "target_team": "Ghana", "line": "4.5"}
        tier, p, _ = slate.resolve_row(row, None, g, None)
        assert tier == "PENDING" and p is None, f"{qt} must stay shadow, got {tier}"

def test_h1_corner_alias_routes_like_1h():
    # team_more_corners_h1 == team_more_corners_1h (string alias) -> founded half-corner
    # route (Pinnacle when present, else measured per-half STOPGAP), NOT shadow.
    g = _team_corners_game()
    a = slate.resolve_row({"question_type": "team_more_corners_h1", "target_team": "Ghana", "line": ""}, None, g, None)
    b = slate.resolve_row({"question_type": "team_more_corners_1h", "target_team": "Ghana", "line": ""}, None, g, None)
    assert a == b and a[0] == "CORNER_HALF_STOPGAP"


def test_period_corners_1h_2h_return_stopgap_floor():
    # team_more_corners_1h/_2h now submit the measured per-half base-rate FLOOR
    # (STOPGAP, not true P) UNCONDITIONALLY — the floor is a constant, available
    # without odds; replaces the proven-wrong CORNER_HALF_SHRINK=0.65.
    g = _team_corners_game()
    for qt, exp in (("team_more_corners_1h", 0.389), ("team_more_corners_2h", 0.410)):
        row = {"question_type": qt, "target_team": "Ghana", "line": "4.5"}
        tier, p, _ = slate.resolve_row(row, None, g, None)
        assert tier == "CORNER_HALF_STOPGAP" and p == exp, f"{qt}: got {tier} {p}"


# --- 3. safe fallback when market absent ------------------------------------

def test_missing_corners_market_falls_back_to_measured_base_rate():
    # NO corners market AND no ladder -> the MEASURED corner base rate (never 0.50/shadow).
    g = {"bookmakers": []}
    row = {"question_type": "team_corners_over", "target_team": "Ghana", "line": "4.5"}
    tier, p, _ = slate.resolve_row(row, None, g, None)
    assert tier == "CORNERS_BASE" and p is not None and 0.02 < p < 0.98

def test_line_not_quoted_uses_ladder_poisson_fit():
    # exact 4.5 not quoted but a 5.5 ladder IS -> Poisson-fit extrapolates DOWN to 4.5
    # (market-derived line-gap read), NOT shadow. P(>=5) must exceed the quoted P(>=6).
    g = _team_corners_game(team="Ghana", line=5.5, n_books=5)   # market at 5.5 only
    row = {"question_type": "team_corners_over", "target_team": "Ghana", "line": "4.5"}  # need 4.5
    tier, p, _ = slate.resolve_row(row, None, g, None)
    assert tier == "CORNERS_LADDER" and p is not None
    cp55 = CN.price_corners_over(g, CN.TEAM_CORNERS_MARKET, "Ghana", 5.5)
    assert p > cp55.p_over   # P(over 4.5) > P(over 5.5), as a CDF must

def test_ladder_fit_multi_line_interpolates():
    # three quoted lines (5.5/6.5/7.5) -> fit Poisson, read P(over 4.5); monotone CDF check
    g = {"bookmakers": [
        _book("B0", CN.TEAM_CORNERS_MARKET, 5.5, -120, 100, team="Ghana"),
        _book("B1", CN.TEAM_CORNERS_MARKET, 6.5, +110, -130, team="Ghana"),
        _book("B2", CN.TEAM_CORNERS_MARKET, 7.5, +220, -260, team="Ghana"),
    ]}
    lp = CN.price_corners_laddered(g, CN.TEAM_CORNERS_MARKET, "Ghana", 4.5)
    assert lp.mapped and 0.02 < lp.p_over < 0.98 and "Poisson-ladder" in lp.note


# --- 4. line mapping: "N or more" == Over (N-0.5) ---------------------------

def test_line_mapping_n_or_more_is_over_n_minus_half():
    # "5 or more" -> line 4.5; the pricer must read the 4.5 line, not 5.5
    g = {"bookmakers": [
        _book("B0", CN.TEAM_CORNERS_MARKET, 4.5, -120, 100, team="Ghana"),
        _book("B1", CN.TEAM_CORNERS_MARKET, 5.5, +110, -130, team="Ghana"),
    ]}
    cp = CN.price_corners_over(g, CN.TEAM_CORNERS_MARKET, "Ghana", 4.5)
    assert cp.mapped and cp.n_books == 1 and abs(cp.line - 4.5) < 1e-9


# --- 5. QUALITY gate (agreement, not book count) ----------------------------

def test_few_agreeing_corners_used_not_discarded():
    # 2 books that AGREE (identical odds, disp 0) -> 'ok' (confident), USED -- NOT discarded
    # for low book count (the old <3-book gate is gone). Count never triggers fallback.
    g = _total_corners_game(line=8.5, n_books=2)
    cp = CN.price_corners_over(g, CN.TOTAL_CORNERS_MARKET, None, 8.5)
    assert cp.mapped and cp.liquidity_flag in ("ok", "thin") and cp.dispersion <= CN.DISPERSION_MAX
    row = {"question_type": "total_corners_over", "target_team": "", "line": "8.5"}
    tier, p, _ = slate.resolve_row(row, None, g, None)
    assert tier in ("CORNERS_OK", "CORNERS_THIN") and p is not None   # routed live, not shadow

def test_single_book_corner_used():
    # ONE book, plausible -> 'thin' (single_book), USED -- a single real read beats shadow
    g = _total_corners_game(line=8.5, n_books=1)
    cp = CN.price_corners_over(g, CN.TOTAL_CORNERS_MARKET, None, 8.5)
    assert cp.mapped and cp.liquidity_flag == "thin" and cp.n_books == 1
    tier, p, _ = slate.resolve_row({"question_type": "total_corners_over", "target_team": "", "line": "8.5"},
                                   None, g, None)
    assert tier in ("CORNERS_OK", "CORNERS_THIN") and p is not None

def test_wide_agreement_corners_used_flagged():
    # books moderately split (0.05 < disp <= 0.10) -> 'thin' (wide_agreement), still USED
    import odds_lib.market_quality as MQ
    g = {"bookmakers": [
        _book("B0", CN.TOTAL_CORNERS_MARKET, 8.5, -160, 130),
        _book("B1", CN.TOTAL_CORNERS_MARKET, 8.5, +120, -150),
        _book("B2", CN.TOTAL_CORNERS_MARKET, 8.5, -110, -110)]}
    cp = CN.price_corners_over(g, CN.TOTAL_CORNERS_MARKET, None, 8.5)
    assert MQ.DISPERSION_OK < cp.dispersion <= MQ.DISPERSION_SCATTER
    assert cp.liquidity_flag == "thin"   # wide_agreement -> still used (not 'low')
    tier, p, _ = slate.resolve_row({"question_type": "total_corners_over", "target_team": "", "line": "8.5"},
                                   None, g, None)
    assert tier in ("CORNERS_OK", "CORNERS_THIN") and p is not None

def _devig_over(over, under):
    from odds_lib.odds import odds_to_prob, remove_vig
    return float(remove_vig(odds_to_prob([over, under], "american"))[0])

def test_sharpness_weighting_tilts_toward_pinnacle():
    # one sharp book (Pinnacle, P(over) lower) + two soft books (higher) -> the weighted
    # estimate is pulled toward the sharp read vs a flat mean.
    g = {"bookmakers": [
        _book("Pinnacle", CN.TOTAL_CORNERS_MARKET, 8.5, +140, -170),
        _book("SoftA", CN.TOTAL_CORNERS_MARKET, 8.5, -160, 130),
        _book("SoftB", CN.TOTAL_CORNERS_MARKET, 8.5, -160, 130)]}
    cp = CN.price_corners_over(g, CN.TOTAL_CORNERS_MARKET, None, 8.5)
    flat = sum(_devig_over(o, u) for o, u in [(140, -170), (-160, 130), (-160, 130)]) / 3
    assert cp.p_over < flat   # sharp (lower) read pulls the weighted estimate below flat mean

def test_scattered_corners_discarded_to_shadow():
    # books DISAGREE wildly -> 'low' -> shadow, regardless of count
    g = {"bookmakers": [
        _book("B0", CN.TOTAL_CORNERS_MARKET, 8.5, -400, 300),
        _book("B1", CN.TOTAL_CORNERS_MARKET, 8.5, +250, -300),
        _book("B2", CN.TOTAL_CORNERS_MARKET, 8.5, -110, -110)]}
    cp = CN.price_corners_over(g, CN.TOTAL_CORNERS_MARKET, None, 8.5)
    assert cp.liquidity_flag == "low" and cp.dispersion > CN.DISPERSION_MAX
    row = {"question_type": "total_corners_over", "target_team": "", "line": "8.5"}
    tier, p, _ = slate.resolve_row(row, None, g, None)
    # scattered exact line is discarded AND excluded from the ladder (agreement filter)
    # -> the MEASURED base rate, not shadow/0.50.
    assert tier == "CORNERS_BASE" and p is not None and 0.02 < p < 0.98


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
