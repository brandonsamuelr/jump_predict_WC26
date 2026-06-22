# Jump Trading / SportsPredict Probability Cup — daily workflow

A small pipeline that turns sportsbook odds into calibrated probabilities for
SportsPredict submissions during the 2026 FIFA World Cup. Three rules: only
market-derived submissions, every step auditable, every output logged.

## Setup (one-time)

1. `.env` in the repo root holds the Odds API key:
   ```
   ODDS_API_KEY=your_key_here
   ```
   (Already gitignored.)
2. Use `.venv/bin/python` directly in commands below, or activate the venv.

## Daily workflow

Five steps. Each one is a single terminal command except step 3 (which is
manual curation in a spreadsheet).

### 1. Fetch fresh odds (bulk + per-event)

```bash
.venv/bin/python scripts/market_sheet.py \
    --forecast-run-id 2026-06-21_morning --fetch --hours 24
```

This makes:

1. One bulk `/odds` call for h2h + totals — **6 credits** at default
   regions (3) × markets (2).
2. One per-event `/events/{id}/odds` call per upcoming match for
   `btts,h2h_h1` — **6 credits per event** at the same regions × 2 markets.
   For a typical 5-game window that's ~30 credits.

Total per fetch: roughly **6 + 6 × N_events** credits. All responses are
cached under `data/raw/`. **This script does not write to
`predictions_log.csv`.** Logging predictions is `submit_sheet.py`'s job
(step 4).

If you also want a full per-book row log of every odds quote (bulk +
per-event), you can additionally run:

```bash
.venv/bin/python scripts/daily_run.py --forecast-run-id 2026-06-21_morning --from-cache
```

(Uses the same caches from step 1, no extra credits, writes to
`data/odds_log.csv`.)

### 2. Auto-stub inventory candidates (free)

```bash
.venv/bin/python scripts/stub_inventory.py --hours 24
```

For each upcoming match this appends candidate rows to
`data/question_inventory.csv`:

- two `team_win` rows (one per team)
- one `both_teams_score` row
- one `halftime_draw` row

All with `include=0` and `source=auto_stub`. Existing rows are never
touched, so re-running is safe. Narrow with `--question-types team_win` if
you don't want the new prop stubs.

### 3. Curate the inventory (manual, ~2 min)

Open `data/question_inventory.csv` in Numbers/Excel.

- For each question that **actually appears on SportsPredict**, set
  `include=1`.
- For any prop question SportsPredict asks that wasn't auto-stubbed (totals,
  BTTS, halftime, etc.), add a new row by hand. Use one of the known
  `question_type` values listed below, or anything unknown will route to
  `ambiguous_review`.

This file is the source of truth for what we plan to submit.

### 4. Build the submission sheet (free) and log

```bash
.venv/bin/python scripts/submit_sheet.py \
    --forecast-run-id 2026-06-21_morning --hours 24 --log
```

This writes `data/submission_sheets/2026-06-21_morning_submit_sheet.csv`
with: the SportsPredict question text, the bet we mapped it to, the
consensus probability, an integer `submit_percent`, `num_books`,
`liquidity_flag`, and a `submit_recommendation` of `submit`,
`review_not_submit`, `review`, or `skip`. With `--log`, the rows
recommended to submit are appended to `data/predictions_log.csv` — that
file is the audit trail of what we intended to submit and why.

**Liquidity gate.** Per-event markets (btts, h2h_h1) often have fewer
books than h2h, so `submit_sheet.py` applies a min-book gate uniformly to
all mapped markets:

| `num_books` | `mapping_status` | `liquidity_flag` | `submit_recommendation` |
|---|---|---|---|
| `>= --thin-books-threshold` (default 5) | `mapped_exact` | `ok` | `submit` (if include=1) |
| `--min-books <= n < --thin-books-threshold` | `mapped_exact` | `thin` | `submit` (if include=1) — eyeball the line manually |
| `< --min-books` (default 3) | `low_liquidity_review` | `low` | `review` (not auto-submitted) |

Override with `--min-books N` and `--thin-books-threshold M` if you want a
stricter or more permissive bar.

### 5. Enter on SportsPredict

For every row where `submit_recommendation = submit`, type the
`submit_percent` value into the matching question on the SportsPredict site.
Skip the rest.

### Mid-day view-only refresh (no API calls, no log writes)

```bash
.venv/bin/python scripts/submit_sheet.py --forecast-run-id 2026-06-21_review --hours 24
```

(Drop `--log` so you don't double-record.) Or use `market_sheet.py` for the
raw market view without the inventory layer.

## What each file is for

```
data/
  raw/                              cached API responses (timestamped JSONs)
                                    bulk:      {sport}__{markets}__{regions}__{ts}.json
                                    per-event: {sport}__event-{event_id}__{markets}__{regions}__{ts}.json
  odds_log.csv                      every (book, outcome) row we've fetched (bulk + per-event)
  question_inventory.csv            SportsPredict questions we plan around
  submission_sheets/                per-run submission sheets (one CSV each)
  predictions_log.csv               audit trail of submit-recommended rows

odds_lib/
  odds.py        odds → probability, vig removal, consensus
  odds_api.py    The Odds API client (bulk + per-event) + JSON parser
  mappings.py    question_type → market dispatch + liquidity gate
  logs.py        CSV append/upsert helpers

scripts/
  fetch_odds.py     thin CLI on the API (list sports / one fetch)
  daily_run.py      logs every fetched odds row, bulk + per-event (no inventory layer)
  market_sheet.py   view-only market consensus for the next N hours
  stub_inventory.py auto-stub team_win + btts + halftime_draw candidates
  submit_sheet.py   THE daily driver: inventory + odds → submission sheet
```

## Supported question types

| `question_type` | Today's behavior | Mapped bet | Source |
|---|---|---|---|
| `team_win` | `mapped_exact` | `h2h / <target_team> / full-time result` | bulk `/odds` |
| `match_total_over` | `mapped_exact` (line defaults to 2.5) | `totals / Over <line> goals` | bulk `/odds` |
| `both_teams_score` | `mapped_exact` | `btts / Yes` | per-event `/odds` |
| `halftime_draw` | `mapped_exact` | `h2h_h1 / Draw` | per-event `/odds` |
| `team_corners_over`, `total_cards_over`, `player_sot_over`, `team_score_2h` | `needs_model` | (specialist markets, deferred) | — |
| `team_offsides_over`, `team_more_fouls`, `team_more_cards`, `penalty_or_red_card` | `unmapped_skip` | (no clean market) | — |
| anything else | `ambiguous_review` | — | — |

Only `mapped_exact` + `include=1` rows are recommended `submit` (and only
if the liquidity gate passes — see step 4). Everything else is shown for
visibility but routed to `review_not_submit`, `review`, or `skip`.

### Per-event markets (now live)

`both_teams_score` and `halftime_draw` are pulled from the per-event
endpoint (`/sports/{sport}/events/{event_id}/odds`) because the bulk
`/odds` endpoint does not surface `btts` or `h2h_h1` for soccer. The cost
is 1 credit per region per market per event; at the default
`btts,h2h_h1` × 3 regions that's 6 credits per upcoming event, ~30
credits for a typical 5-game window. The min-book liquidity gate (see
step 4) is the safety net against thin or one-book per-event quotes.

### Inventory examples for the new types

```
include,match,question_type,target_team,target_player,line,sports_predict_question,source,notes
1,Spain vs Saudi Arabia,team_win,Spain,,,Will Spain win the match?,manual,
1,Spain vs Saudi Arabia,match_total_over,,,2.5,Will the match have 3 or more total goals?,manual,
1,Belgium vs Iran,team_win,Belgium,,,Will Belgium win the match?,manual,
1,Belgium vs Iran,match_total_over,,,2.5,Will the match have 3 or more total goals?,manual,
```

The `line=2.5` for `match_total_over` is the canonical "3 or more goals"
mapping; if blank, the mapper defaults to 2.5.

## Strategy seam

The submission sheet stores `market_prob` and `submit_prob` separately. Today
`submit_prob = market_prob` and `strategy_name = market_consensus_v1`. When a
future strategy nudges submissions (public-bias adjustment, contest position,
etc.), it changes `submit_prob` and `strategy_name`; `market_prob` stays as
the immutable record of what the books said.

## Credit budget

The Odds API tier is 20,000 credits/month. Both endpoints cost **1 credit
per region per market**; per-event also multiplies by the number of events
in the fetch window. Current defaults:

- Bulk `--markets h2h,totals --regions us,uk,eu` → **6 credits per fetch**.
- Per-event `--per-event-markets btts,h2h_h1 --regions us,uk,eu` →
  **6 credits per upcoming event**. For 5 events ≈ **30 credits**.
- Combined per fetch ≈ **36 credits**. Two fetches/day for a 32-day
  tournament ≈ **~2,300 credits**, well under the 20k cap.

If you want to expand the per-event market set later (team totals, cards,
corners — priorities 3-5 from the sprint plan), the only changes needed
are: (a) a new `question_type` in [mappings.py](odds_lib/mappings.py),
(b) optionally a new auto-stub in [stub_inventory.py](scripts/stub_inventory.py),
(c) the new market key added to `--per-event-markets`. The fetch/cache
plumbing already handles arbitrary per-event market lists.
