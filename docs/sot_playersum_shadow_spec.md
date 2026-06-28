# Player-SOT-sum shadow estimator — spec

**Status:** SHADOW / DIAGNOSTIC ONLY. Never production. Promote a route only if it beats the
incumbent OOS on a date-split *with a high-λ slice* (see Disposition).

## Purpose
Team SOT is the one attacking family with **no market anywhere** (no team-SOT, total-SOT, or
2H-SOT line on any book or exchange — verified 2026-06-27). Today the family is model-only:
`team_sot_over` rides the logistic ([sot_count_model.py](../odds_lib/sot_count_model.py)); the
comparison / 2H / totals / both-SOT routes ride the concave map
([rate_layer.py](../odds_lib/rate_layer.py) `team_sot_mu`, `μ(λ)=A(1−e^{−Bλ})`).

This instrument builds an **independent second estimate** of team SOT *level* by summing the
λ recovered from the **player-SOT props we already pull** (`player_shots_on_target`, de-vigged
via `player_prop_pricing.price_player_prop` → `market_prob_vig_adjusted`):

```
λ_player = −ln(1 − p_devig(player ≥1 SOT))
propsum_raw(team) = Σ_{propped players} λ_player        # a biased LOWER BOUND (coverage gap)
playersum_μ(team) = propsum_raw × coverage_factor(λ-band)   # corrected estimate
```

It then compares `playersum_μ` against the concave-map `team_sot_μ(λ)` for the same team, and —
when the match resolves — calibrates **both** against realized team SOT. The two estimates are
*differently sourced* (player props vs 1X2+totals), so divergence is informative; convergence is
only *partly* informative (both ultimately reflect the same match's attacking expectation —
agreement is partly self-confirming, **outcomes are the arbiter**).

## What it can and cannot detect

This instrument compares **expected SOT (the level/mean)**. That is exactly one dimension.

| concern | can this instrument see it? |
|---|---|
| concave-map μ **too high/low at high λ** (level bias) | **YES** — its target use |
| coverage / vig-inversion error (mid-λ) | partly — shows up as divergence but is *not* the level question |
| count distribution **over-dispersion** (fat tail vs Poisson) — *flaw 3* | **NO — STRUCTURALLY BLIND** |

### GUARDRAIL 1 — tag every divergence by λ-regime
A divergence is only diagnostic of **level bias** in the **high-λ band**, because that is where the
concave map's saturation is load-bearing. In the **mid-λ band** a divergence is dominated by
coverage truncation + one-sided-vig inversion noise — it answers nothing about the map.

Every logged row carries `lambda_band ∈ {low, mid, high}` (defaults: `low<1.0`, `1.0≤mid<1.8`,
`high≥1.8`, tunable). **Read divergences within-band.** The level-bias question is answered by the
**high-band** divergence trend, *not* the pooled mean. (A confound to watch: if `coverage_factor`
itself drifts with λ, a high-band divergence could be coverage, not level — hence the coverage
factor is fit **per λ-band**, and a flat/insufficient-data band is logged as such, not silently
pooled.)

### GUARDRAIL 2 — this instrument is blind to under-dispersion (flaw 3); never read agreement as "map is fine"
Both estimates convert a mean to a probability through a **Poisson** tail. If realized team SOT is
**over-dispersed** (variance > mean, fat upper tail), *both* will understate `P(SOT ≥ high count)`
**by the same mechanism** — so they will **agree with each other and both be wrong**.

Therefore the decision table for the open SOT-tail calibration question is:

```
playersum_μ vs concave_μ:   AGREE      AND  outcome high-tail calibration:  OK
        → level is validated; tail concern was likely noise.

playersum_μ vs concave_μ:   DISAGREE (high band)
        → level bias candidate; inspect concave saturation; this is the route's job.

playersum_μ vs concave_μ:   AGREE      AND  outcome high-tail calibration:  MISSES HIGH
        → CONCLUDE: the problem is SHAPE (over-dispersion), NOT level,
          and THIS INSTRUMENT WAS BLIND TO IT.
          Do NOT conclude "the map is fine." The fix is a dispersion model
          (NB/Conway-Maxwell on the count), which this tool cannot motivate or validate.
```

The script prints this banner on every run so the conclusion is never quietly inverted.

## Coverage factor (measured, gated — not typed)
`coverage_factor[band] = E[team SOT] / E[propsum_raw]` for matches in that band. It is a **measured,
refreshable, OOS-gated parameter** (`data/models/sot_coverage_factor.json`), same category as
`H1_SHARE` / the offside pooled rate — **never** a typed constant, and **never** used to shrink an
output toward anything (it reshapes a market-sourced sum; the result still moves fully with the
per-match propsum, k=1). Fit from either StatsBomb position-level SOT (share of team SOT from the
propped set) or accumulated live (propsum_raw vs realized team SOT).

**Until the factor is fit + passes its gate, the instrument runs DIAGNOSTIC-ONLY:** it logs
`propsum_raw`, `concave_μ`, the raw ratio `R = propsum_raw/concave_μ`, and `lambda_band` — no
corrected level, no claims. A declining `R` across the high band (with low/mid as controls) is the
earliest level-bias signal available before the factor exists.

## Disposition
- **Do not swap** the player-sum into production. It carries coverage truncation, one-sided-vig
  λ-inversion, and (for tails) the same Poisson blindness as the incumbent.
- Log `playersum_μ`, `concave_μ`, `logistic P` (for `team_sot_over` thresholds), `lambda_band`,
  and realized team SOT to `data/models/sot_playersum_shadow.csv`.
- Feed large within-band divergence into the **market-quality agreement gate** (downgrade
  confidence / flag for inspection) — *not* into the price.
- **Promote** a player-sum route to production ONLY if it beats the incumbent (concave / logistic)
  on **OOS outcome-Brier**, date-split, **with a dedicated high-λ slice** — never on pooled fit,
  never on crowd agreement.
