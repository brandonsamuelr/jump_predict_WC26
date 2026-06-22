# Player-Prop Truth-Model Plan

Status: **planning / not built**. Until a model here meets the minimum
criteria below, player props with no direct market route to
`player_prop_review_required` (see `odds_lib/decision_engine.py`). Lineup
status/role are **features and risk tags only** — never a hard-coded
`p_truth`.

This document scopes the *first alpha module*: a data-derived `p_truth` for
player props, starting with `player_sot_over` and `player_goal_or_assist`.
The pattern proven here generalizes to other prop families later. We are not
abandoning high-volume market/derived/shadow execution; this is additive.

---

## 0. Why player props first

Recent slates suggest the crowd badly misprices player props when starting
status, role, and likely minutes are ignored (e.g. a benched/absent attacker
the crowd still rates highly). That gap — between `p_field` (crowd) and
`p_truth` (reality) — is a plausible, recurring edge. The constraint: the
edge must be **data-derived or market-derived**, not a hand-built table.

Keep `p_field` and `p_truth` separate at all times. Lineup info may move
`p_truth` a lot while barely moving `p_field`; that divergence is the alpha.

---

## 1. Targets

| Question type            | Target variable (binary)                              | Notes |
|--------------------------|-------------------------------------------------------|-------|
| `player_sot_over`        | P(player records > `line` shots on target in match)   | line usually 0.5 (i.e. 1+ SOT) |
| `player_sot_2h_over`     | P(player records > `line` SOT in 2nd half)            | secondary; depends on minutes |
| `player_goal_or_assist`  | P(player scores OR assists in match)                  | |
| `player_goal`            | P(player scores in match)                             | |

All targets are conditional on **appearance and minutes**, which is why
lineup status / expected minutes are first-class features, not truth values.

Decompose where useful:
`P(event) = P(plays) * P(event | minutes, role, matchup)`. A clean model can
predict the conditional rate and fold in an appearance/minutes model.

---

## 2. Candidate features (pre-lock only)

### `player_sot_over`
- starting status (`starter` / `bench_*` / `out_of_squad` / `unknown`)
- expected minutes (if known)
- position / role
- team implied goals (from totals + supremacy — see §4, currently missing)
- opponent defensive strength
- player shots per 90
- player SOT per 90
- recent starts / minutes (rolling form)
- competition / national-team context
- the `line`
- direct player-prop market price, if available (becomes `p_truth` directly)

### `player_goal_or_assist`
- starting status
- expected minutes
- position / role
- team implied goals
- player goals per 90
- assists per 90
- xG per 90, xA per 90 (if available)
- penalty / set-piece role (if available)
- recent starts / minutes
- opponent strength
- direct player-prop market price, if available

---

## 3. Data we currently have

- **Contest history**: `data/historical/sportspredict_collected_data.csv` —
  58 resolved player-prop rows (31 `player_sot_over`, 13
  `player_goal_or_assist`, 9 `player_sot_2h_over`, 5 `player_goal`). Carries
  `field_prob` (locked crowd), `result`, `submitted_percent`, `actual_rbp`,
  `line`. **Gap**: `target_player` is largely blank in these rows, so they
  are usable for crowd/field calibration but weak for player-specific truth.
- **Lineup files**: `data/lineups/*.json` — manual, status + role +
  optional `expected_minutes`. Only exist for matches we prepared.
- **Odds**: per-event caches in `data/raw/` give h2h / totals / btts →
  match context (favorite, implied totals, BTTS). Player-prop markets from
  The Odds API are sparse for the WC (US-book coverage gaps).
- **Feature dataset builder**: `scripts/build_player_prop_features.py` →
  `data/models/player_prop_feature_rows.csv`. Produces the pre-lock feature
  block + separated eval columns. This is the spine the model trains on.

## 4. Data we are missing (must source before modeling)

- **Per-player rate stats**: shots/90, SOT/90, goals/90, assists/90,
  xG/90, xA/90. Not in any current file.
- **`target_player` on historical rows**: needed to join rates to outcomes.
- **Team implied goals**: derivable from totals line + h2h supremacy, but
  not yet computed — left `None` in the feature builder rather than faked.
- **Opponent strength**: needs a team-rating source.
- **Expected minutes**: only present when manually entered.
- **Recent starts/minutes**: needs a match-log history per player.

---

## 5. Possible data sources (evaluate later; do NOT scrape without sign-off)

- FBref / StatsBomb-style event + per-90 summaries
- FotMob / Sofascore / ESPN match logs
- Kaggle / open football datasets
- Bookmaker player-prop markets where available (best `p_truth`: a real price)
- Manual CSV collection for the specific contest players each slate

Manual CSV collection is the most realistic near-term path: for the ~2
players per match the contest actually asks about, hand-collect per-90 rates
and recent minutes into a small joinable table. Low volume, high signal.

---

## 6. Baselines

A model must beat these to earn live use:

1. **Crowd/field baseline** (`p_field`): the engine's semantic-anchor +
   historical blend. This is what we'd submit anyway.
2. **Role/status heuristic** (the *removed* hard-coded table): kept only as a
   conceptual baseline to beat — it must NOT be re-enabled in production.
3. **Market baseline**: where a direct player-prop price exists, that is the
   truth and no model is needed.

---

## 7. Validation method

- **Unit of evaluation**: realized contest RBP (relative Brier vs field),
  not raw accuracy — that is what the contest pays.
- **Split**: temporal / grouped by slate (no leakage across a match's
  questions). Mirror the field-model audit that found V2 ≈ ridge so we hold
  this model to the same bar.
- **Leakage rule**: `result`, `actual_rbp`, locked `crowd_percent`,
  `submitted_percent` are **evaluation-only**. They live after the feature
  block in the dataset and must never enter X. The dataclass
  `PlayerPropFeatures` contains pre-lock fields *only*, by construction.
- **Metrics**: mean RBP vs field, Brier, calibration curve, and
  hit-rate on the subset where lineup status disagrees with crowd pricing
  (the hypothesized edge).

---

## 8. Minimum criteria before live use

A player-prop truth model may set `p_truth` in live recommendations only when
ALL hold:

1. Trained / empirically calibrated on real data (not hand-set constants).
2. Beats the crowd/field baseline on **grouped, out-of-sample** RBP by a
   margin that survives the slate-level noise (same discipline that kept the
   field model out of production).
3. Calibrated (reliability curve close to diagonal) on held-out data.
4. Has a `p_truth_source` recognized as production-grade by
   `is_truth_source_production_grade` (`trained_model` /
   `empirical_calibrated_model`). Heuristic sources stay capped at
   `HEURISTIC_TRUTH_CONFIDENCE_CAP` and cannot drive aggressive submissions.
5. Documented failure modes + a kill-switch (allowlist-style gate, like the
   `STRONG_HISTORICAL_ALLOWLIST` pattern).

Until then: `decision_mode = player_prop_review_required`, `p_truth = None`,
`needs_manual_review = True`.

---

## 9. Build sequence (proposed)

1. Extend `build_player_prop_features.py` to compute `team_implied_goals`
   from totals + supremacy (real derivation, not a constant).
2. Stand up the manual per-player rate CSV + a join into the feature rows.
3. Backfill `target_player` on historical rows where recoverable.
4. Fit a simple, interpretable baseline (logistic / GBM) for
   `player_sot_over` first; evaluate per §7.
5. Only if it clears §8, wire a `trained_player_prop_model` truth source into
   the engine behind an explicit allowlist. Generalize the pattern to
   `player_goal_or_assist`, then other prop families.
