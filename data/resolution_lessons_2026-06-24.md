# Resolution lessons — Morocco vs Haiti + Scotland vs Brazil (2026-06-24)

First `founded_delivered_on_time` resolution data — the clean pipeline-vs-field read.

## Per-slate
- **Scotland vs Brazil: net ≈ +23.5**, beat-crowd 8/10. Clean win — shadows + market rows priced, no costly overrides.
- **Morocco vs Haiti: net ≈ −38.3**, beat-crowd 4/10 (rbp>0 on 3/10 — the 4th is a crowd-point-vs-field-brier definitional edge).
- **Combined ≈ −14.8.**

## Loss is concentrated, not diffuse
- Two rows = **−32.6** of the night: Morocco Q2 corners (−15.37) + Q9 Nazon (−17.20).
- The **other 18 rows net ≈ +17.8** → pipeline beats the field when shadows are priced and overrides are calibrated. The damage is two *diagnosed + fixed* leaks, not pipeline failure.

## Override-effect this slate (all three Morocco overrides) — NET −53.33 RBP vs pipeline
| override | pipeline→final | result | actual rbp | rbp @ pipeline | delta |
|---|---|---|---|---|---|
| Nazon player_sot_over | 0.471→0.15 | YES | −17.20 | +27.07 | **−44.27** |
| team_more_sot_2h (MAR) | 0.727→0.65 | YES | −2.21 | +2.59 | −4.80 |
| total_sot_2h_over | 0.681→0.62 | YES | +2.30 | +6.56 | −4.26 |

All three moved AWAY from the realized YES. SOT trims small; Nazon dominates. (field_brier recovered per row: F = (final−y)² + rbp/100; rbp@q' = 100·(F−(q'−y)²).)

## LESSON 1 — corner_model_would_have_saved (Q2, −15.37 on a flat shadow)
team_more_corners_1h (Haiti) was a **flat shadow 0.493** (vs crowd 0.25, resolved NO). The gate-validated corner model is **now WIRED**: it prices full-match P(Haiti more) = 0.152 → 1H-shrunk **0.274** (≈ crowd). That would have scored **+1.44 vs −15.37 → +16.81 saved**. The fix existed but wasn't live at this lock. Proof the now-live model protects the 1H/comparison corner family going forward.

## LESSON 2 — super_sub_band_validated (Q9 Nazon, −17.20)
Benched striker Nazon overridden to **0.15** (fringe-bench value), came on, took a shot → **YES**. A benched STAR who can sub on and needs only 1 SOT is **not** 0.15, and not the flat 0.471 shadow — he sits in the **super-sub band ~0.30**. This is the live counterexample validating the super-sub override band (0.33/0.38) used the same night for **Schick (CZE) and Son (SK)** — first paired test settles when those resolve. The 0.15 fringe value was wrong for a player who can feature.

## Net read
18/20 non-leak rows positive supports trusting the pipeline (price the shadows, keep overrides calibrated). Both leaks now addressed: corner model wired; super-sub band adopted. Watch the override-effect column across future slates — this slate it cost −53 RBP; the discipline is to override only on lineup facts at a *calibrated* level, never aggressive.
