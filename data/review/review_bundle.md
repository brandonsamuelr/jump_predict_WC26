# WC26 Probability Cup — Data Review Bundle

A self-contained snapshot for an external reviewer. Pair this file with the raw
CSVs listed at the bottom. Generated from the live pipeline; numbers reflect 5
slates logged (4 resolved + Portugal pending).

---

## 1. Context — how scoring works (read first)

Contest: World Cup binary forecasting, ~10 yes/no questions per match, submit a
probability 0–100% per question. Scored by **Relative Brier Points (RBP)**.

- The field benchmark is the **mean of individual Briers** (not the Brier of the
  crowd mean), which by Jensen equals `Var(q_i) + (c̄ − y)²`. We gain RBP when our
  Brier beats that benchmark, lose when it's worse.
- Recovery identity used throughout: `field_Brier = (final − y)² + actual_rbp/(100·m)`,
  then any candidate q' is scored `RBP(q') = 100·m·(field_Brier − (q'−y)²)`.
- The **optimal submission for any row is our best estimate of the true prob θ.**
  The crowd is *evidence about θ*, never a target. The realized crowd is revealed
  only at lock, so it is **post-lock diagnostic only** — never a pre-lock input.

**Our submission rule (every row):**
```
p_submit = c_hat + k · (p_model − c_hat)        clipped to [0.02, 0.98]
```
- `c_hat` = pre-lock field proxy (historical qt-mean / computed type base rate / global mean).
- `p_model` = our independent model probability (None on no-model rows).
- `k` = per-(source-class, subtype) edge multiplier. At the current tiny sample
  every class is FROZEN on a **structural prior** (MARKET/ENGINE 0.90, PROP-confirmed
  0.75, PROP-thin 0.40, SOT-comparison 0.60, SOT-single 0.50, SOT-total_2h 0.50,
  SHADOW 0.00). It sharpens toward a data-fitted least-squares `k` as resolved
  match-clusters accumulate.

**Source classes:** MARKET (de-vigged sharp odds) · ENGINE (market-anchored goals
model) · PROP (player props; only on confirmed starters) · RATE_SOT (shots-on-target
model) · SHADOW (no independent edge → submit c_hat).

**Competitive state:** ~148 RBP, rank ~684 / 3307. Gap to #100 ≈ 857. To close it
over ~60 remaining matches (~600 weighted-questions) we must out-score #100 by
**~+1.43 RBP/weighted-question**. Whether the strategy achieves that is the open
empirical question; the pace tracker (report 4) measures it forward.

---

## 2. Data dictionary — `measurement_log.csv` (the core dataset)

One row per submitted question. Probabilities are decimals (0–1); RBP/crowd are
contest display units.

| column | meaning |
|---|---|
| `run_id` | slate id (date) |
| `match` | "Home vs Away" (the cluster unit — questions in a match share one game script, so they are correlated) |
| `question_number` | Q1–Q10 |
| `question_type` | canonical type (e.g. `team_win`, `total_sot_2h_over`, `compound_btts_over_2_5`, `player_goal`) |
| `tier` | pipeline resolution: `MARKET`, `ENGINE_GOALS`, `RATE_SOT`, `RATE_SOT_CMP`, `PROP_ok`, `PROP_thin`, `PENDING` (=shadow) |
| `weak_row_class` | free-text row annotation; `FLAG…` entries are watch-notes (no separate notes column exists) |
| `source` | who set `final_submitted`: `pipeline` \| `manual` \| `llm` |
| `p_hat` | **raw model** probability `p_model` (blank for shadow / no-model rows) |
| `shadow` | **`c_hat`**, the pre-lock field proxy |
| `manual_estimate`, `llm_estimate` | human / LLM estimates where they exist (never imputed as 0) |
| `pipeline_submit` | what the optimizer produced (counterfactual if overridden) |
| `final_submitted` | what was actually locked (basis of `actual_rbp`) |
| `override_reason` | free text; categorized at report time by `classify_override` into soft / hard_qa / other / rounding |
| `multiplier` | contest question weight (all 1 so far) |
| `result` | `Yes`/`No` (blank until the match resolves) |
| `actual_rbp` | realized RBP vs the field (blank until resolved) |
| `crowd_prob` | realized crowd % (integer; **post-lock diagnostic, not a pre-lock input**) |
| `if_yes_rbp`, `if_no_rbp` | RBP swing under each outcome (from the locked sheet) |
| `at_stake` | swing magnitude `|if_yes − if_no|` |

**Reading discipline baked into the analysis:** effective evidence ≈ **match-cluster
count (4)**, not row count (39). A conclusion only counts if it holds across matches.
Fitted `k` is shrunk toward the prior in squared-deviation units and FROZEN until a
class has both ≥5 clusters and ≥4 "active" rows (|p_model − c_hat| > 0.05).

---

## 3. What we'd most value an outside opinion on

1. **Is the edge-weighted rule sound for RBP, and are the 0.90 MARKET/ENGINE priors
   right?** Fitted (unfrozen) `k_hat` is MARKET 0.61, ENGINE 1.92 — i.e. the data so
   far hints MARKET should shrink *more* and ENGINE *less* than 0.90, but on only 4
   clusters. Are we over- or under-expressing edge?
2. **SOT rows may not be earning their keep.** In the tier scorecard, `RATE_SOT`
   `rbp_final` (0.69) is *below* `rbp_shadow` (1.74) — though the *pipeline* SOT
   (3.49) beats shadow, so the gap is largely manual overrides. Plus a recurring
   "SOT model reads high vs crowd & LLM" signal (today's Portugal Q8/Q9). Should more
   SOT rows just shadow?
3. **Override leakage:** soft manual overrides run −3.21 RBP/q (report 2). We've
   disabled the soft category by default. Agree?
4. **`total_sot_2h_over` dispersion audit (report 5):** mean slope is data-backed but
   the Poisson tail is overdispersed (var/mean ≈ 2.6), so we keep the level (recenter)
   and discount the tail steepness via k=0.50. Is that the right decomposition?
5. **Small-sample risk:** 4 clusters. Are we reading anything as signal that's noise?

---

## 4. Report 1 — Per-class edge table (`scripts/edge_report.py`)

```
DELIVERABLE 1 — per-class edge table  (24 resolved rows, 4 match clusters)
                             n  clusters  n_active  eff_n_k  sum_d2  k_prior    k_hat  k_shrunk  frozen  k_deployed            confidence  mean_rbp_final  mean_rbp_model  mean_rbp_base
source_class source_subtype
ENGINE       engine          8         4         8     4.21  0.2632     0.90    1.919     1.561    True        0.90  LOW(prior-dominated)            2.90            2.78          -5.93
MARKET       market          7         4         2     1.87  0.1123     0.90    0.613     0.842    True        0.90  LOW(prior-dominated)            1.78            1.57           1.91
PROP         confirmed       1         1         1     1.00  0.0086     0.75    3.871     1.097    True        0.75  LOW(prior-dominated)            4.10            5.96          -1.57
             thin            1         1         0     1.00  0.0000     0.40  -88.167     0.390    True        0.40  LOW(prior-dominated)            7.29            6.76           7.40
RATE_SOT     comparison      1         1         1     1.00  0.1537     0.60    1.383     0.687    True        0.60  LOW(prior-dominated)            7.70            7.41         -19.72
             single          3         3         2     1.80  0.1062     0.50    0.309     0.462    True        0.50  LOW(prior-dominated)           -2.08           -1.37           1.68
             single_2h       1         1         0     1.00  0.0000     0.50  126.000     0.504    True        0.50  LOW(prior-dominated)           -0.48           -0.79           0.44
             total_2h        2         2         2     1.99  0.0692     0.50    2.025     0.805    True        0.50  LOW(prior-dominated)            5.43           12.92           2.46

k_hat = UNCLIPPED fit (may be <0 = anti-predictive); k_deployed = clip(k_shrunk,[0,1]).
eff_n_k / n_active = the REAL fitting sample (rows where the model took a position) — governs trust, not n.
Fit on c_hat (pre-lock shadow), NOT realized crowd. Every class is FROZEN on its prior at this cluster count (by design).
```

## 5. Report 2 — Tier / match scorecard (`scripts/measure.py`)

```
39 of 50 logged rows resolved, across 4 match(es).

=== MATCH scorecard — totals (the honest unit of evidence) ===
                         n  rbp_final  beat_field  rbp_pipeline  rbp_shadow  pipe_vs_shadow  pipe_vs_final  n_man  rbp_manual  pipe_vs_manual  n_llm  rbp_llm  pipe_vs_llm
Argentina vs Austria  10.0       43.2        0.70          42.0        15.4            26.6           -1.2    0.0         NaN             NaN    0.0      NaN          NaN
France vs Iraq         9.0       32.3        0.78          44.7       -32.3            77.0           12.4    9.0        33.0            11.7    0.0      NaN          NaN
Jordan vs Algeria     10.0       25.3        0.90          22.3        22.5            -0.3           -3.0    1.0         1.0            -2.6   10.0     -2.8         25.1
Norway vs Senegal     10.0       18.4        0.60          34.1        25.6             8.5           15.7   10.0        17.0            17.1   10.0     19.0         15.1
ALL                   39.0      119.2        0.74         143.1        31.3           111.8           23.9   20.0        51.0            26.2   20.0     16.3         40.1

=== TIER scorecard — per-question means ===
                 n  rbp_final  beat_field  rbp_pipeline  rbp_shadow  pipe_vs_shadow  pipe_vs_final  n_man  rbp_manual  pipe_vs_manual  n_llm  rbp_llm  pipe_vs_llm
ENGINE_GOALS   8.0       2.90        0.75          2.78       -5.93            8.71          -0.12    4.0        3.48            1.80    3.0    -5.36         6.77
MARKET         7.0       1.78        0.71          1.57        1.91           -0.34          -0.21    3.0        1.82           -2.46    4.0     0.82        -0.36
PENDING       15.0       4.02        0.80          4.58        4.58            0.00           0.56    8.0        3.27            0.86    9.0     3.45        -1.93
PROP_ok        1.0       4.10        1.00          5.96       -1.57            7.53           1.86    1.0        4.10            1.86    1.0     6.10        -0.14
PROP_thin      1.0       7.29        1.00          6.76        7.40           -0.64          -0.53    0.0         NaN             NaN    0.0      NaN          NaN
RATE_SOT       6.0       0.69        0.50          3.49        1.74            1.76           2.80    4.0        0.33            4.42    3.0    -2.69        12.92
RATE_SOT_CMP   1.0       7.70        1.00          7.41      -19.72           27.13          -0.29    0.0         NaN             NaN    0.0      NaN          NaN
ALL           39.0       3.06        0.74          3.67        0.80            2.87           0.61   20.0        2.55            1.31   20.0     0.81         2.01

rbp_final=what we submitted; rbp_pipeline=trust-the-optimizer; rbp_shadow=always-shadow.
pipe_vs_*>0 => pipeline beat that strategy (paired). Effective evidence ~ 4 match-clusters, NOT 39 rows.
```

## 6. Report 3 — Override leakage (`scripts/override_report.py`)

```
DELIVERABLE 2 — OVERRIDE LEAKAGE  (leak = rbp_final - rbp_pipeline; <0 = override hurt)
50 logged rows | 11 real overrides (|dev|>0.011) | 10 of those resolved & scorable

--- leakage by category (resolved overrides) ---
          n  mean_leak  total_leak
other     2      -0.07       -0.15
soft      8      -3.21      -25.65

ALL resolved overrides: n=10  mean_leak=-2.58/q  total=-25.8 RBP
  SOFT subset (the disabled category): n=8  mean_leak=-3.21/q  total=-25.6 RBP

--- every real override (audit the category labels) ---
                 match   q                 type category  pipeline  final    dev   leak   reason
  Argentina vs Austria  Q5             team_win    other     0.641   0.66  0.019   1.33   (none)
  Argentina vs Austria  Q8        team_sot_over    other     0.669   0.68  0.011  -1.48   (none)
        France vs Iraq  Q1 team_more_corners_h1     soft     0.405   0.35 -0.055   4.15   role_sensitive_shadow_favorite_should_control_territory
        France vs Iraq  Q2      team_more_fouls     soft     0.535   0.45 -0.085  -8.63   role_sensitive_shadow_favorite_possession_foul_risk
        France vs Iraq  Q3         team_card_2h     soft     0.493   0.57  0.077  -8.19   underdog_defending_or_chasing_second_half_card_risk
     Norway vs Senegal  Q1   team_offsides_over     soft     0.485   0.47 -0.015  -1.57   moderate_shadow_toward_manual_but_keep_close
     Norway vs Senegal  Q3  penalty_or_red_card     soft     0.389   0.35 -0.039   2.88   split_difference_physical_match_but_shadow_likely_high
     Norway vs Senegal  Q5    total_sot_2h_over     soft     0.814   0.57 -0.244 -15.03   high_threshold_sot_model_risk_..._manual_cap
     Norway vs Senegal Q10          player_goal     soft     0.233   0.27  0.037  -1.86   star_usage_..._do_not_overtrust_intuition
     Jordan vs Algeria  Q6   player_sot_2h_over     soft     0.478   0.45 -0.028   2.60   modest_human_trim_2h_player_sot_felt_high_..._mahrez_starting
Portugal vs Uzbekistan Q10          player_goal    other     0.360   0.19 -0.170    NaN   confirmed_bench_but_proven_supersub_..._crowd_underpriced (PENDING)

POLICY: SOFT category disabled by default in apply_override (allow_soft to force); HARD_QA kept.
CAVEAT: leak correlated within a match — read totals across the 4 clusters, not as independent rows.
```

## 7. Report 4 — Forward pace tracker (`scripts/pace_tracker.py`)

```
DELIVERABLE 6 — FORWARD PACE TRACKER
latest (2026-06-23, seed_live): rank 684 of 3307 | our_rbp=148 | #100=1006 | #1=1498
static gaps: to #100 = 857 RBP | to #1 = 1350 RBP
to close #100 over ~60 matches (~600 weighted-q) we must OUT-score
  #100 by +1.43 RBP/weighted-q (and #1 by +2.25).

only the baseline snapshot exists — forward pace appears after the next snapshot
(add one after each slate resolves; this is the metric that says if the strategy works).
```

## 8. Report 5 — `total_sot_2h_over` slope/dispersion audit (`scripts/audit_total_sot_2h_slope.py`)

```
[1] lambda_total over 20 calibration matches (REAL): mean=2.746 sd=0.307 range 2.36–3.37 (narrow)
[2] LEVEL check (full-match total SOT, unpaired means): observed 10.78 vs model 10.34 -> GOOD (mean not biased)
[3] DISPERSION test: mean=10.78 var=27.69 -> dispersion index 2.57 (Poisson=1.0).
    lambda-heterogeneity explains only ~0.86 of the variance, so the tail is genuinely overdispersed
    (corroborates team-level R^2=0.37). Poisson plug-in P(>=4) is OVER-confident at the extremes.
[4] across-tempo p_submit (c_hat=0.623): mid-tempo lands ~0.64 for ANY k (level robust);
    extremes are where k matters (cagey k=0.5 -> 0.44, high k=0.5 -> 0.73).
[5] DECISION: keep the recenter (mean slope is data-backed); discount the tail via k=0.50 as an
    EXPLICIT overdispersion discount (magnitude not calibratable yet, n=9). recenter moves the MEAN,
    k shrinks the TAIL — different objects, so trusting one and discounting the other is consistent.
```

---

## 9. Raw files to attach alongside this bundle

| file | why |
|---|---|
| `data/measurement_log.csv` | the core dataset (50 rows, 5 matches) — see §2 |
| `data/historical/sportspredict_collected_data.csv` | field/crowd base rates by question type (basis of every c_hat) |
| `data/historical/sportspredict_submitted_scoring_rows.csv` | earlier scored history |
| `data/leaderboard_snapshots.csv` | competitive standing + forward-pace inputs |
| `data/models/sot_calibration_rows.csv` | SOT-vs-λ calibration inputs (relevant to the SOT-bias question) |

Optional, for a view on *method* not just data: `odds_lib/edge.py`, `optimizer.py`,
`measurement.py`, `field_model.py` (+ `rate_layer.py`, `match_engine.py` for the models).

**Do not send** `.env` (API key), `data/raw/*.json` (large odds dumps, no signal).
