# StatsBomb open-data feasibility probe (2026-06-25)

**Question:** which placeholder/period question families can we build & validate from FREE StatsBomb event data, and on how many real international matches?

**Answer:** the half-split STRUCTURE wing is fully buildable for free; team-specific minnow tendencies are not (no minnows, no odds).

## What's there
Senior men's internationals: **333 matches total, 314 in the modern full-detail era** —
WC 2022 (64), WC 2018 (64), Euro 2024 (51), Euro 2020 (51), AFCON 2023 (52), Copa America 2024 (32).
(+19 legacy WC 1958–1990, sparse 1–6 each, low value.) Event-level: every shot/foul/corner/card/offside with `period` + `minute` + `team`.

## Confirmed field paths (checked against real events, not assumed)
- **SOT**: `type.name=='Shot'` → `shot.outcome.name` ∈ {Goal, Saved, Saved to Post} = on target; {Off T, Post, Wayward, Blocked} = off. ⚠ StatsBomb "Blocked" ≠ Opta SOT — pick a convention.
- **Fouls**: `type.name=='Foul Committed'` (+ `team.name`, `period`).
- **Corners**: `type.name=='Pass'` AND `pass.type.name=='Corner'` (NOT a top-level type).
- **Cards**: `foul_committed.card.name` + `bad_behaviour.card.name` (Yellow/Second Yellow/Red).
- **Offsides**: `type.name=='Pass'` AND `pass.outcome.name=='Pass Offside'` (~5/match); team = the side caught offside. (No top-level "Offside" type in this data.)
- **Half-split**: `period` (1=1H, 2=2H, 3/4=ET, 5=shootout) — clean buckets; `minute` on all events.

## Buildable (free, on ~314 intl matches)
CORNER_HALF_SHRINK, H1_SHARE (re-confirm 0.44), 1H/2H SOT comparison, cards 2H/period — all BUILDABLE.
Offsides = PARTIAL (derivable but no odds for context). Fouls = PARTIAL (universal half-split yes; minnow tendency no).

## Hard limits
No odds (structure corpus, applied on top of live market numbers). Tournaments only — minnows (Haiti etc.) absent. Paid Opta/Sportmonks would only add minnow team-tendencies, which API-Football already showed are sparse.

Files: `probe_summary.json` (machine-readable), `_competitions_raw.json`, `_wc2022_matches.json`, `_counts.json`, `_eventcounts.json`.
