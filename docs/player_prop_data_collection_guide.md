# Player-Prop Data Collection Guide

How to fill `data/models/player_prop_collection_template.csv` for an upcoming
slate. This raw collected data is the input to research and, eventually, a
data-derived `p_truth` model (see `player_prop_truth_model_plan.md`). It is
**not** a model and contains **no invented probabilities**.

## Generating a template

```bash
# seed rows for every match with a lineup file in data/lineups/
python scripts/build_player_prop_collection_template.py

# one match, custom path (keep collected files out of the default path)
python scripts/build_player_prop_collection_template.py \
    --match "New Zealand vs Egypt" --out data/models/2026-06-22_nz_egy_collection.csv

# header-only
python scripts/build_player_prop_collection_template.py --empty
```

The seeder fills only **known facts** (match, teams, player, question_type,
and the lineup status/role/expected_minutes/source copied verbatim from the
lineup JSON). Everything else is blank for you to collect.

## The three hard rules

1. **Never invent a number.** Unknown probability, expected minutes, or rate
   → leave the cell **blank**. Blank = "not collected", which is honest and
   trainable-around. A guessed value silently poisons the dataset.
2. **Record provenance.** Every manually entered value gets a `*_source`
   and/or an `entry_reason`. "FBref season per90", "official team sheet @
   25min pre-KO", "DraftKings prop @ 18:00" — enough to re-find/audit it.
3. **No leakage.** The evaluation columns (crowd %, submitted %, result, RBP)
   are **post-lock**. They are for scoring/calibration only and must never be
   used as model features. They live at the end of the file for that reason.

## Column dictionary

### Identity (who/what)
| column | meaning |
|---|---|
| `collection_id` | stable key `{game_date}_{match_slug}_{player}_{qt}` (auto) |
| `game_date` | match date (YYYY-MM-DD) |
| `match` | "Home vs Away" |
| `competition` | e.g. "World Cup group stage" (manual) |
| `target_team` | the player's team |
| `opponent_team` | the other team (auto when teams parse) |
| `target_player` | player name as SportsPredict asks it |
| `question_type` | `player_sot_over`, `player_goal_or_assist`, … |
| `question` | exact SportsPredict wording, if known |
| `line` | prop line (e.g. 0.5 for "1+ SOT") |

### Lineup / availability — **auto-filled from lineup JSON; verify before trusting**
| column | meaning |
|---|---|
| `lineup_status` | `starter` / `bench_high_usage` / `bench_low_usage` / `bench_unknown` / `out_of_squad` / `unknown`. Use `out_of_squad` ONLY when the bench is confirmed; otherwise a non-XI player is `bench_unknown`. |
| `lineup_role` | `central_attacker` / `wide_attacker` / `attacking_midfielder` / `central_midfielder` / `defender` / `goalkeeper` / `unknown` |
| `expected_minutes` | **blank unless genuinely known** (e.g. manager quote, confirmed sub plan). Never estimate. |
| `lineup_source` | where the XI/status came from |
| `lineup_captured_at` | when the lineup was observed (lineups change pre-KO) |

### Recent form — manual / data-derived
| column | meaning |
|---|---|
| `recent_starts_last5` | starts in the player's last 5 team matches |
| `recent_minutes_last5` | minutes over the last 5 |
| `recent_form_source` | provenance |
| `recent_form_reason` | free text if judgement was involved |

### Player rates (per 90 unless noted) — data-derived
| column | meaning |
|---|---|
| `shots_per90` / `sot_per90` | shots / shots-on-target per 90 |
| `goals_per90` / `assists_per90` | goals / assists per 90 |
| `xg_per90` / `xa_per90` | expected goals / assists per 90 (if available) |
| `is_penalty_taker` | true/false/blank |
| `setpiece_role` | corners / free_kicks / none (manual) |
| `rates_sample_matches` | how many matches the rates cover (context for noise) |
| `rates_source` | e.g. FBref/StatsBomb/manual |

### Match / market context — market-derived
| column | meaning |
|---|---|
| `target_team_win_prob` | from h2h |
| `team_implied_goals` | **derive** from totals + supremacy; do not guess. Blank until derived. |
| `match_total_line` | the totals line used |
| `match_total_over_2_5_prob` | from totals |
| `btts_prob` | both-teams-to-score Yes |
| `opponent_strength` | rating or note |
| `market_context_source` | cache file / book / timestamp |

### Direct player-prop market — best `p_truth` when it exists
| column | meaning |
|---|---|
| `has_direct_prop_market` | true/false/blank |
| `direct_prop_market_prob` | de-vigged prob if a real prop market is found |
| `prop_market_source` | book + timestamp |

### Evaluation-only (post-lock) — NEVER features
| column | meaning |
|---|---|
| `crowd_percent` | locked crowd YES % |
| `submitted_percent` | what we submitted |
| `result` | Yes / No |
| `actual_rbp` | realized relative Brier points |
| `if_yes_rbp` / `if_no_rbp` | conditional RBP shown pre-result |

### Provenance / meta
| column | meaning |
|---|---|
| `entered_by` | who collected the row |
| `entry_reason` | why any manual/judgement value was chosen |
| `notes` | anything else |

## Workflow per slate

1. Add/confirm lineup files in `data/lineups/` for the slate.
2. Generate a dated template to a **non-default** path so you don't overwrite
   it on the next run.
3. Fill identity gaps (`question`, `line`, `competition`).
4. Collect rates + recent form from your chosen source; fill `*_source`.
5. Pull market context from the odds caches; fill `market_context_source`.
6. After matches resolve, fill the evaluation columns from the contest.
7. Append completed rows into a persistent research dataset for modeling.
