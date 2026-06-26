# Market-availability audit — "what books price that we never request"

Date: 2026-06-24 (probe event: **Colombia vs DR Congo**, kickoff 02:00Z, us/uk/eu)
Scope: AUDIT + CLASSIFICATION ONLY. No `REFRESH_MARKETS` change, no pricing change.

## How this was done (cheaply)
- The Odds API does **not** expose a "list markets for sport" endpoint, and the
  per-event response only contains markets you explicitly request. So the catalog
  cannot be discovered for free — you must name candidate keys.
- Key cost lever: on the **per-event** `/events/{id}/odds` endpoint you are billed
  for **markets actually returned**, not requested
  (`cost = unique_markets_returned × regions`). So I requested a broad superset of
  *confirmed-valid* soccer keys (valid-only, to avoid a 422 that would void the call)
  against one upcoming event and only paid for what exists.
- Valid key list taken from The Odds API betting-markets docs; fouls/offsides keys
  are **not offered for soccer** by the API, so they were not probed (would error).

### Credits
- Successful probe: **87 credits** (29 markets returned × 3 regions). Remaining: **19,114**.
- A first attempt billed the same ~87 but crashed before saving (cache filename too
  long — the market list is encoded in the filename). **Total audit spend ≈ 174.**
- Per-event liquidity for markets we already fetch came from the existing
  Panama-vs-Croatia cache (free, 0 credits).

Per-event liquidity scale used below (book count, us/uk/eu): **Liquid ≥10 · Moderate 5–9 · Thin 3–4 · Ultra-thin 1–2.**

---

## 1. Full available-market catalog

### Markets we ALREADY fetch (REFRESH_MARKETS) — per-event books (Panama cache)
| key | books | type |
|---|---|---|
| h2h | 49 | match result |
| totals | 23 | match goals O/U (2.5–3.5) |
| btts | 17 | both teams score |
| h2h_h1 | 11 | 1H result |
| spreads | 11 | match handicap |
| player_goal_scorer_anytime | 10 | player goal |
| totals_h1 | 7 | 1H goals O/U |
| player_shots_on_target | 4 | player SOT |
| team_totals | 4 | team goals O/U |
| player_to_score_or_assist | 2 | player score-or-assist |

### NEW markets that exist but we never request (this probe) — per-event books
**Corners** (the big find — none ever fetched):
| key | books | notes |
|---|---|---|
| alternate_totals_corners | 9 (Moderate) | total corners O/U, lines 3.5→13.5+ |
| alternate_team_totals_corners | 5 (Moderate) | team corners O/U, lines 1.5→10.5 |
| corners_1x2 | 4 (Thin) | which team more corners (incl. tie/Draw) |
| alternate_spreads_corners | 3 (Thin) | corners handicap |

**Cards**:
| key | books | notes |
|---|---|---|
| alternate_totals_cards | 6 (Moderate) | total cards O/U, lines 1.5→4.5 — **card-count convention caveat** |
| alternate_spreads_cards | 1 (Ultra-thin) | cards handicap |
| player_to_receive_card | 4 (Thin) | player booked |
| player_to_receive_red_card | 1 (Ultra-thin) | player red card |

**Half / period (for 2H derivations)**:
| key | books | notes |
|---|---|---|
| alternate_totals_h1 | 10 (Liquid) | 1H goals O/U (wider lines than totals_h1) |
| alternate_totals_h2 | 5 (Moderate) | **2H goals O/U**, lines 0.5→3.5 |
| totals_h2 | 1 (Ultra-thin) | 2H goals O/U (single book) |
| h2h_h2 | 4 (Thin) | **2H result** (= which team scores more in 2H) |
| h2h_3_way_h2 | 3 (Thin) | 2H 3-way |
| h2h_3_way_h1 | 6 (Moderate) | 1H 3-way |
| team_totals_h1 | 4 (Thin) | **team 1H goals O/U** (0.5, 1.5) |
| btts_h1 | 6 (Moderate) | both teams score in 1H |
| spreads_h1 | 6 (Moderate) | 1H handicap |
| double_chance_h1 | 2 (Ultra-thin) | 1H double chance |
| halftime_fulltime | 6 (Moderate) | HT/FT combo |
| **spreads_h2 / team_totals_h2** | **0** | *requested, not returned for this event* |

**Match-level extras** (alternatives to what we have):
| key | books | notes |
|---|---|---|
| h2h_3_way | 46 (Liquid) | 3-way result |
| alternate_totals | 16 (Liquid) | match goals O/U at **any** line (fills lines `totals` lacks) |
| draw_no_bet | 12 (Liquid) | DNB |
| alternate_spreads | 12 (Liquid) | Asian handicap → margin distribution |
| double_chance | 11 (Liquid) | double chance |
| alternate_team_totals | 5 (Moderate) | team goals O/U at wider lines |

**Player extras** (no current question maps, but available):
| key | books | notes |
|---|---|---|
| player_first_goal_scorer | 10 (Liquid) | first scorer |
| player_shots | 3 (Thin) | player total shots, lines 0.5→5.5 |
| player_last_goal_scorer | 3 (Thin) | last scorer |
| player_assists | 2 (Ultra-thin) | player assist O/U 0.5 |

**Confirmed NOT offered by the API for soccer** (genuinely unpriced): fouls, offsides,
team/match shots-on-target aggregates, team/match total-shots aggregates, penalties,
match "any red card", any 2H corners/cards markets.

---

## 2. Question-type → classification table (priority-sorted)

Legend: handling = current pipeline tier · class = DIRECT / DERIVABLE / NO MARKET.

### A. TOP PRIORITY — currently SHADOW/FLOOR, a DIRECT market exists
| question_type | current | class | market(s) | books | upgrade | notes |
|---|---|---|---|---|---|---|
| total_corners_over | shadow | **DIRECT** | alternate_totals_corners | 9 | **Y** | Over at contest line (9+ = O8.5). Scope ✓ full-match corners. Cost the −4.77 Ghana row. |
| team_corners_over | shadow | **DIRECT** | alternate_team_totals_corners | 5 | **Y** | Over at team line (5+ = O4.5). Scope ✓. |
| total_cards_over | needs_model/shadow | **DIRECT** | alternate_totals_cards | 6 | **Y** | ⚠ verify card-count convention (cards vs booking points; red=1 or 2) before trusting. |
| team_more_corners_full | shadow | **DIRECT** | corners_1x2 | 4 | **Y (thin)** | P(team) from 3-way; tie = Draw outcome — confirm question treats a tie as "no". |

### B. SECOND PRIORITY — currently ENGINE/RATE (model-only), a DIRECT market can ANCHOR it
| question_type | current | class | market(s) | books | upgrade | notes |
|---|---|---|---|---|---|---|
| total_goals_2h_over / second_half_goals_over | ENGINE_GOALS | **DIRECT** | alternate_totals_h2 | 5 | **Y** | 2H goals O/U direct (2+ = O1.5). Also DERIVABLE = totals − totals_h1. |
| team_more_goals_2h | ENGINE_GOALS | **DIRECT** | h2h_h2 (or h2h_3_way_h2) | 4 (3) | **Y** | 2H result IS "which team scores more in 2H"; P(team) = its 2H-win price (tie=Draw). |
| team_score_1h | ENGINE_GOALS | **DIRECT** | team_totals_h1 @0.5 | 4 | **Y** | Over 0.5 = team scores in 1H. |
| team_score_any | ENGINE_GOALS | **DIRECT** | team_totals @0.5 | 4 | (Y) | Already fetched; team Over 0.5. Engine currently; could market-anchor. |
| team_total_goals_over | (engine/unwired) | **DIRECT** | team_totals | 4 | (Y) | Already fetched but may not be wired to this qtype. |
| match_total_over/under (non-2.5 line) | MARKET | **DIRECT+** | alternate_totals | 16 | (Y) | `alternate_totals` fills any contest line `totals` doesn't carry. |

### C. DERIVABLE but THIN/weak — handle with floor/thin discipline or leave
| question_type | current | class | market(s) | books | upgrade | notes |
|---|---|---|---|---|---|---|
| team_more_cards | shadow | DERIVABLE | alternate_spreads_cards @0 | 1 | marginal | cards handicap sign → P(more cards); 1 book = unreliable. Effectively keep shadow. |
| team_score_2h | ENGINE | DERIVABLE | team_totals − team_totals_h1 | 4/— | weak | team_totals_h2 not returned this event; derivation needs both halves present. |
| second_half_more_goals | ENGINE | DERIVABLE | alt_totals_h1 vs alt_totals_h2 | — | weak | no joint; O/U lines alone don't give P(2H>1H) cleanly. Keep engine. |

### D. NO MARKET — genuinely unpriced; stays shadow / candidate for the slow model project
| question_type | current | class | why |
|---|---|---|---|
| team_offsides_over | shadow | NO MARKET | API offers no offsides market for soccer. |
| team_more_fouls | shadow | NO MARKET | API offers no fouls market for soccer. |
| penalty_or_red_card / penalty_awarded | shadow | NO MARKET | no penalty or match-level red-card market (player_to_receive_red_card is per-player, 1 book). |
| team_sot_over / team_sot_2h_over / total_sot_2h_over / match_total_sot_over / second_half_total_sot_over / team_more_sot_2h | RATE_SOT | NO MARKET | no team/match shots-on-target aggregate market (only player-level SOT). |
| both_teams_sot_1h / _h1_1plus / _2h_1plus | shadow | NO MARKET | depends on team SOT, which is unpriced. |
| player_sot_2h_over | shadow | NO MARKET | no 2H player-SOT market. |
| team_card_2h / total_cards_2h_over | shadow | NO MARKET | no 2H (or 1H) cards market — only full-match cards. |
| team_more_corners_h1 / team_more_corners_2h / second_half_corners_over | shadow | NO MARKET | no period (1H/2H) corners markets — only full-match corners. |
| team_first_goal_2h / team_score_first_2h | shadow | NO MARKET | no "first goal of 2H" market. |
| compound_btts_over_2_5 / compound_first_goal_score_2h | ENGINE | NO single market | joint/correlated event; no direct AND market. Engine is the right tool. |

### E. Already DIRECT (no change — for completeness)
team_win (h2h) · match_total_over/under (totals) · both_teams_score (btts) ·
halftime_team_win / halftime_team_winning / halftime_team_lead / halftime_draw (h2h_h1) ·
player_goal (player_goal_scorer_anytime) · player_sot_over (player_shots_on_target) ·
player_goal_or_assist (player_to_score_or_assist + anytime floor).

### F. New markets with NO current question (note for future contest rows)
player_first_goal_scorer (10) · player_assists (2) · player_to_receive_card (4) ·
player_shots (3) · draw_no_bet (12) · double_chance (11) · alternate_spreads / Asian
handicap → margin distributions (12).

---

## 3. Summary

Counting **currently shadowed/floored** rows (the upgrade-value population):

- **DIRECT upgrades available: 4** — `total_corners_over`, `team_corners_over`,
  `total_cards_over`, `team_more_corners_full`. (Plus a 5th, `team_more_cards`, is
  technically derivable but 1-book/unreliable.)
- **DERIVABLE (weak/thin): ~2** — `team_more_cards`, `team_score_2h` (conditional on
  both-half markets posting).
- **Genuinely NO MARKET: ~17** — all offsides/fouls/penalty rows, every team/match
  SOT row, all 2H/1H cards & corners period rows, "first goal of 2H", and the
  team-SOT-dependent both-teams-SOT rows.

Bonus, beyond shadow rows: **4 ENGINE rows can be market-anchored** —
`total_goals_2h_over`, `team_more_goals_2h`, `team_score_1h`, `team_score_any` — moving
them from pure model to a market-calibrated price.

### Headline
The corners markets are the clear win: `alternate_totals_corners` (9 books) and
`alternate_team_totals_corners` (5 books) turn the two corners rows we've been conceding
to the crowd into clean market prices. `alternate_totals_cards` (6 books) does the same
for total cards **pending a card-counting-convention check**. The 2H goal markets
(`alternate_totals_h2`, `h2h_h2`, `team_totals_h1`) let several engine rows be
market-anchored. Everything fouls/offsides/penalty/SOT-aggregate is confirmed unpriced
by the API → those correctly stay shadow / the slow-model project.

### Discipline notes carried forward (do before any implementation)
1. **Liquidity**: all new markets are Moderate–Thin (1–9 books), well below h2h/totals.
   Apply the same gate as score-or-assist (min_books, thin flag); 1-book markets
   (`alternate_spreads_cards`, `totals_h2`) are not trustworthy.
2. **Scope checks** (same rigor as score-or-assist):
   - Cards: confirm books count *cards* the way the contest does (yellow+red as count,
     red as 1 vs 2, "booking points" vs card count). Mismatch here is worse than shadow.
   - corners_1x2 / 2H-result comparisons: confirm tie handling matches the question.
3. These were probed on ONE event; per-event availability varies (e.g. spreads_h2 /
   team_totals_h2 didn't post here). Any wiring must degrade safely to shadow when a
   market is absent for a given match — same PENDING fallback the pipeline already uses.
