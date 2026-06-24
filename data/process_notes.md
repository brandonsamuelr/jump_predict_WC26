# Process notes — durable operational lessons

## Read-back verification at submission (instituted 2026-06-24, after the Panama entry-shift)

**Rule:** before locking a slate, compare the contest's **YOU** column against the
code/refresh output **position-by-position (Q1..Q10)** before submitting.

**Why:** on 2026-06-24 (Panama vs Croatia), a one-row upward shift in the
entry/recording handoff corrupted Q1–Q4 — the pipeline output the correct intended
values but the contest recorded shifted values (e.g. Q2 intended 0.288 → scored 0.49
on stake 44; Q1 0.485 → 0.29 on stake 30). The model/pipeline was correct; the error
was pure transcription into the contest UI. A position-by-position read-back at lock
catches this class of error before it scores.

**How these are logged:** `source=entry_error`, `override_reason` starts `entry_shift_…`
(→ category `entry_shift` via `classify_override`), `pipeline_submit` = the intended/code
value, `final_submitted` = the scored value. So the cost is `RBP(intended) − RBP(scored)`
= `rbp_pipeline − rbp_final` per row, and these rows are **excluded from strategy
analysis** (recording errors, not model decisions).
