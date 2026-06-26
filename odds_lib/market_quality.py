"""Unified market-quality policy (the law): book COUNT never triggers a fallback.

A real bookmaker line is a live, match-specific estimate of true P; a constant/shadow
is conditioned on no match. So a market read is NOT used ONLY when:
  (1) there is NO market (zero books) -> the sole surviving count branch,
  (2) the de-vigged output is IMPLAUSIBLE (stale/erroneous/broken derivation), or
  (3) the books genuinely SCATTER (they disagree beyond the scatter cliff).
Present books -> SHARPNESS-WEIGHTED estimate, used when plausible. "thin"/book-count is a
diagnostic LABEL only; it never changes the value or triggers a fallback.

UNIFIED dispersion policy (ONE pair of thresholds for EVERY family):
  disp <= DISPERSION_OK (0.05)         -> books AGREE        -> use (flag 'confident')
  DISPERSION_OK < disp <= SCATTER(0.10) -> moderately split  -> USE, flag 'wide_agreement'
  disp > DISPERSION_SCATTER (0.10)     -> genuine scatter    -> FALL BACK (flag 'scattered')
  n == 1 (no dispersion to compute)    -> single real read   -> use if plausible ('single_book')

*** INTERIM thresholds — NOT yet calibrated against resolved-row accuracy. ***
Justification from normal market noise: observed clean reads disperse <=0.016 (so 0.05 is
~3x headroom, won't false-reject good thin reads); genuine scatter in testing hit ~0.18,
and 0.10 is the cliff where books disagree by ~10 prob-points = real uncertainty. TODO
(also in memory): measure whether high-dispersion reads were actually LESS accurate on
resolved rows and move the cliff to where accuracy degrades.
"""

from __future__ import annotations

import statistics as _stats

DISPERSION_OK = 0.05        # <= this: books agree (confident)
DISPERSION_SCATTER = 0.10   # > this: genuine scatter -> fall back

# Sharp-book reference: Pinnacle is THE canonical sharp anchor; Circa / BetOnline /
# Bookmaker.eu / Betfair Exchange are also sharp. Coarse tiers (weights are confidence
# multipliers for the weighted mean, NOT pricing constants). Unknown books = neutral 1.0.
_SHARP_WEIGHTS = {
    "pinnacle": 3.0,
    "circa": 2.0, "circa sports": 2.0,
    "betonline.ag": 2.0, "betonline": 2.0,
    "bookmaker.eu": 2.0, "bookmaker": 2.0,
    "betfair": 2.0, "betfair ex": 2.0, "betfair exchange": 2.0, "smarkets": 2.0,
}
_DEFAULT_WEIGHT = 1.0


def book_weight(title: str) -> float:
    """Sharpness weight for a book title (coarse: sharp >1, recreational/unknown = 1)."""
    t = (title or "").strip().lower()
    if t in _SHARP_WEIGHTS:
        return _SHARP_WEIGHTS[t]
    for k, w in _SHARP_WEIGHTS.items():       # substring (titles vary: "Pinnacle Sports")
        if k in t:
            return w
    return _DEFAULT_WEIGHT


def weighted(reads: list[tuple[str, float]]):
    """reads = [(book_title, prob), ...]. Returns (weighted_mean, dispersion, n) or None.
    dispersion = UNWEIGHTED stdev of the raw per-book probs (the agreement signal)."""
    if not reads:
        return None
    ws = [book_weight(t) for t, _ in reads]
    ps = [p for _, p in reads]
    wm = sum(w * p for w, p in zip(ws, ps)) / sum(ws)
    disp = _stats.pstdev(ps) if len(ps) > 1 else 0.0
    return wm, disp, len(ps)


def quality_flag(dispersion: float, n: int) -> str:
    """confident | wide_agreement | scattered | single_book."""
    if n <= 1:
        return "single_book"
    if dispersion <= DISPERSION_OK:
        return "confident"
    if dispersion <= DISPERSION_SCATTER:
        return "wide_agreement"
    return "scattered"


def is_scatter(dispersion: float, n: int) -> bool:
    return n >= 2 and dispersion > DISPERSION_SCATTER


# Map the unified quality flag onto the legacy {ok, thin, low} liquidity_flag that the
# slate routing already understands (low -> caller falls back; ok/thin -> used, both k=1).
# confident -> ok; wide_agreement/single_book -> thin (used, less certain); scatter -> low.
def liquidity_flag(dispersion: float, n: int) -> str:
    q = quality_flag(dispersion, n)
    return {"confident": "ok", "wide_agreement": "thin",
            "single_book": "thin", "scattered": "low"}[q]


# --- per-family PLAUSIBILITY bands (sanity bounds on the de-vigged/derived OUTPUT,
#     NOT submitted values). A read outside its band = stale/broken -> distrust. -------
# goal 1H share: derived from the 106k-match HTHG/HTAG distribution (aggregate 0.4394,
#   per-match expected-share central range). The others are degenerate-quote catches
#   (a de-vigged O/U or comparison prob shouldn't be ~0 or ~1 for a real WC fixture);
#   INTERIM, to be tightened from each family's resolved distribution.
PLAUSIBLE_BANDS = {
    "h1_share":    (0.30, 0.60),   # 106k-match goal half-share distribution
    "over_under":  (0.02, 0.98),   # de-vigged O/U prob: degenerate/stale catch
    "comparison":  (0.05, 0.95),   # more-X 3-way de-vig: balanced match never this extreme
}


def in_band(value: float, family: str) -> bool:
    lo, hi = PLAUSIBLE_BANDS.get(family, (0.0, 1.0))
    return lo <= value <= hi


__all__ = ["DISPERSION_OK", "DISPERSION_SCATTER", "book_weight", "weighted",
           "quality_flag", "is_scatter", "liquidity_flag", "PLAUSIBLE_BANDS", "in_band"]
