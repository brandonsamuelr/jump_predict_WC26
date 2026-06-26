# SGO settlement corpus (model-fit fuel)

Source: SportsGameOdds Rookie **free trial** (api.sportsgameodds.com/v2). Harvested 2026-06-25.
One-time grab so we can FIT conditional half-stat models, then DROP SGO (see reference_provider_survey).

- **538 fixtures**, recent (2026-04-15 .. 2026-06-09), 8 leagues:
  INTERNATIONAL_SOCCER (129, incl. WC/friendlies) + EPL/UCL/Bundesliga/LaLiga/SerieA/Ligue1/MLS.
- **favorite_gap: 510/538 (95%)** — `fg_source`: sgo_devig_ml (406, de-vigged 2-way moneyline across books)
  or sgo_fairodds (104, SGO no-vig fallback) or none (28). favorite_gap = 2*P(home DNB) - 1.
- **Box-score stats: ~76%** of fixtures (results{} populated). Columns per side×period (home/away × 1h/2h/game):
  cornerKicks, yellowCards, offsides, shots_onGoal, fouls, shots, points, possessionPercent.
  **Per-half (1h/2h) present** — this is the half-split fuel OddsPapi markets don't give.
- **~390 fixtures have BOTH favorite_gap AND box-score stats** = the fittable set.

## What this founds (later model-fit task, NOT done here)
- **offsides** (the ONLY founding path anywhere) — offsides counts + favorite_gap -> conditional model.
- **half-SOT / half-corners / cards** — per-half share/comparison models (cross-check vs OddsPapi Pinnacle reads).

## ABSENCES (honest)
- **redCards / penalties: NOT in SGO settlement** -> penalty / pen-or-red stays UNFOUNDABLE (base-rate anchor).
- team identities (home/away short codes) are JOIN/AUDIT keys ONLY — never a model feature (transfer-filter discipline).
- Stats are TEAM-level (not per-player) -> does NOT found player-SOT (that stays on The Odds API prop + lineup gate).
