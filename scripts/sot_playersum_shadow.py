"""Player-SOT-sum SHADOW estimator (diagnostic only — never production).

Builds an INDEPENDENT second estimate of team SOT *level* by summing the lambda recovered from
the player-SOT props we already pull, and compares it to the concave map (rate_layer.team_sot_mu).
Populated at lock when props post; calibrated against realized team SOT as matches resolve.

Spec + rationale: docs/sot_playersum_shadow_spec.md

TWO GUARDRAILS (baked in here, not optional):
  1. Every divergence is tagged with its lambda_band. A divergence is diagnostic of LEVEL bias
     only in the HIGH band; in the MID band it is coverage/vig-inversion noise. Read within-band.
  2. This instrument compares MEANS through a Poisson tail, so it is STRUCTURALLY BLIND to
     under-dispersion (flaw 3). If player-sum and concave AGREE but outcomes still miss the high
     tail, the conclusion is "the problem is SHAPE, not level, and this tool was blind to it" --
     NEVER "the map is fine." The banner below prints on every run to prevent that inversion.

    .venv/bin/python scripts/sot_playersum_shadow.py            # banner + self-test
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odds_lib.rate_layer import team_sot_mu  # the concave map we are cross-checking

SHADOW_CSV = Path("data/models/sot_playersum_shadow.csv")
COVERAGE_PATH = Path("data/models/sot_coverage_factor.json")

# lambda regimes (tunable). HIGH is the only band where a divergence answers the level question.
BANDS = (("low", 0.0, 1.0), ("mid", 1.0, 1.8), ("high", 1.8, float("inf")))

BLINDNESS_BANNER = (
    "\n  !! BLIND TO UNDER-DISPERSION (flaw 3): this tool compares MEANS via a Poisson tail.\n"
    "     If player-sum == concave BUT outcomes miss the high tail -> the problem is SHAPE\n"
    "     (over-dispersion), and this instrument could NOT see it. Do NOT read agreement as\n"
    "     'the map is fine'. Read divergence only WITHIN the high lambda_band.\n"
)


# --- pure core (unit-testable) ---------------------------------------------------------------

def lambda_from_prop(p_devig: float) -> float:
    """Recover a player's expected SOT from a de-vigged P(>=1 SOT): lambda = -ln(1-p)."""
    p = min(max(float(p_devig), 1e-6), 1 - 1e-9)
    return -math.log(1.0 - p)


def playersum_raw(devig_probs) -> float:
    """Sum of player lambdas over the propped set. A biased LOWER BOUND on team SOT (coverage gap)."""
    return float(sum(lambda_from_prop(p) for p in devig_probs if p is not None))


def lambda_band(lam: float, bands=BANDS) -> str:
    for name, lo, hi in bands:
        if lo <= lam < hi:
            return name
    return bands[-1][0]


def load_coverage_factor(path: Path = COVERAGE_PATH) -> dict | None:
    """Per-band coverage factors {band: factor, ...} once fit+gated. None -> diagnostic-only mode."""
    try:
        d = json.loads(path.read_text())
        return d if d.get("oos_gate", {}).get("beats_floor") and d.get("factors") else None
    except Exception:
        return None


def coverage_correct(raw: float, band: str, table: dict | None) -> float | None:
    """raw propsum -> corrected team-SOT level. None if no gated factor for this band (then we stay
    diagnostic-only -- we do NOT invent a factor, and we never shrink toward a constant)."""
    if not table:
        return None
    f = (table.get("factors") or {}).get(band)
    return float(raw * f) if f else None


def diagnose(team: str, lam: float, devig_probs, n_covered: int,
             realized_sot=None, table: dict | None = None) -> dict:
    """One team-match diagnostic row. concave_mu is the incumbent level; playersum is the
    independent cross-check. Divergence is meaningful only when read inside its lambda_band."""
    raw = playersum_raw(devig_probs)
    concave = team_sot_mu(lam)
    band = lambda_band(lam)
    corrected = coverage_correct(raw, band, table)
    return {
        "team": team,
        "lam": round(lam, 4),
        "lambda_band": band,
        "n_covered": n_covered,
        "propsum_raw": round(raw, 4),
        "concave_mu": round(concave, 4),
        "ratio_raw_over_concave": round(raw / concave, 4) if concave else None,
        "playersum_mu_corrected": round(corrected, 4) if corrected is not None else None,
        "divergence_corrected_minus_concave": (
            round(corrected - concave, 4) if corrected is not None else None),
        "mode": "corrected" if corrected is not None else "diagnostic_only",
        "realized_team_sot": realized_sot if realized_sot is not None else "",
    }


# --- logging ---------------------------------------------------------------------------------

FIELDS = ["team", "lam", "lambda_band", "n_covered", "propsum_raw", "concave_mu",
          "ratio_raw_over_concave", "playersum_mu_corrected",
          "divergence_corrected_minus_concave", "mode", "realized_team_sot"]


def record(rows: list[dict], path: Path = SHADOW_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


# --- self-test / banner ----------------------------------------------------------------------

def _selftest():
    # lambda inversion round-trips
    assert abs(lambda_from_prop(1 - math.exp(-1.3)) - 1.3) < 1e-6
    assert lambda_band(0.7) == "low" and lambda_band(1.4) == "mid" and lambda_band(2.4) == "high"
    # a heavy favourite (high band): props for 5 attackers ~0.45 each -> raw lower bound < concave
    probs = [0.55, 0.5, 0.45, 0.4, 0.35]
    d = diagnose("Demo FC", lam=2.3, devig_probs=probs, n_covered=len(probs), table=None)
    assert d["lambda_band"] == "high" and d["mode"] == "diagnostic_only"
    assert d["propsum_raw"] < d["concave_mu"]                 # truncation -> raw is a lower bound
    assert d["ratio_raw_over_concave"] < 1.0
    print("self-test OK")
    print(f"  demo (no coverage factor -> diagnostic-only): {d}")


if __name__ == "__main__":
    print(__doc__.splitlines()[0])
    cov = load_coverage_factor()
    print(f"\ncoverage factor: {'LOADED (corrected mode)' if cov else 'absent -> DIAGNOSTIC-ONLY mode'}")
    print(BLINDNESS_BANNER)
    _selftest()
    print("\nThis is a populate-at-lock instrument: feed de-vigged player-SOT props + engine lambda\n"
          "per team via diagnose()/record() when R32 props post; add realized_team_sot as matches\n"
          "resolve, then calibrate BOTH estimates against outcomes (high band especially).")
