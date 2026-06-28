"""R32 SHADOW for the tiered prop de-vig (NOT a tuning target).

AUTO-CAPTURE, so the outcome-validation accumulates without a manual call on a busy
lock night (the whole point: it must answer "does tiered beat flat ON OUTCOMES").

  - capture_pending(...)  is called at LOCK (scripts/pregame_refresh.py) for every
    market-priced STARTER prop: it stores the tiered output + provenance + what each
    flat candidate would produce, keyed by (match, question_number). Last write wins,
    so re-running the refresh just refreshes the snapshot. Bench/sub/unmapped rows are
    skipped (their market read assumes starter minutes -> would add noise).
  - harvest(...)          is called at REVIEW (scripts/measure.py): it joins pending
    snapshots to resolved outcomes in the measurement log and appends finalized rows
    to the shadow. Idempotent (dedupes by match+question_number).

The R32 question is "does the TIERED logic beat ANY flat constant on outcomes?" -- NOT
"which flat constant fits best" (that relapses into the n=9 SOT trap). Provenance
(overround_source + favorite/longshot band) makes the tier-a-vs-tier-b selection-bias
check auditable instead of silent.

    .venv/bin/python -m odds_lib.prop_devig_shadow     # harvest + evaluate
"""
from __future__ import annotations

import csv
from pathlib import Path

from .player_prop_pricing import (
    SHADOW_OVERROUND_CANDIDATES, PropPricing, price_player_prop)

PENDING_CSV = Path("data/models/prop_devig_pending.csv")
SHADOW_CSV = Path("data/models/prop_devig_shadow.csv")
LOG_PATH = Path("data/measurement_log.csv")
KEY = ("match", "question_number")

FIELDS = (["match", "question_number", "target_player", "question_type", "y", "raw",
           "overround_source", "overround_used", "overround_prior_n",
           "tiered_p", "prop_band"]
          + [f"p@{c}" for c in SHADOW_OVERROUND_CANDIDATES])


def candidate_probs(raw: float) -> dict[float, float]:
    """What each flat candidate overround would produce from the same raw."""
    return {c: round(raw / c, 4) for c in SHADOW_OVERROUND_CANDIDATES}


def prop_band(p: float) -> str:
    """Favorite/longshot bucket for the selection-bias audit (tier-b under-estimates
    margin on longshots, where two-sided quotes are rarer)."""
    return "favorite" if p >= 0.55 else ("longshot" if p < 0.35 else "mid")


def _row_fields(pricing: PropPricing, match: str, question_number, y) -> dict | None:
    """Build a FIELDS dict from a resolved/priced PropPricing. None if no recoverable
    raw (un-mapped / no market). ``y`` may be 0/1 or "" (pending, outcome unknown)."""
    raw = pricing.market_prob_raw
    if raw is None or pricing.market_prob_vig_adjusted is None:
        return None
    row = {
        "match": match, "question_number": question_number,
        "target_player": pricing.target_player or "",
        "question_type": pricing.question_type, "y": y, "raw": round(raw, 4),
        "overround_source": pricing.overround_source,
        "overround_used": pricing.overround_used,
        "overround_prior_n": pricing.overround_prior_n,
        "tiered_p": pricing.market_prob_vig_adjusted,
        "prop_band": prop_band(pricing.market_prob_vig_adjusted),
    }
    row.update({f"p@{c}": v for c, v in candidate_probs(raw).items()})
    return row


def shadow_row(pricing: PropPricing, match: str, y: int, question_number="") -> dict | None:
    """Resolved shadow row (PropPricing + outcome). Kept as the explicit API."""
    return _row_fields(pricing, match, question_number, int(y))


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def _is_market_starter_prop(tier: str | None) -> bool:
    """Capture only props we actually price off the market for a STARTER (PROP_ok /
    PROP_thin / PROP_direct_thin / PROP_proxy_floor). PROP_SUB (minutes-scaled bench)
    and PENDING are excluded -- their outcome would not test the market de-vig."""
    return bool(tier) and tier.startswith("PROP_") and tier != "PROP_SUB"


def capture_pending(match: str, row: dict, game_json: dict, tier: str,
                    path: Path = PENDING_CSV) -> bool:
    """At LOCK: snapshot the tiered de-vig + candidates for one market-priced starter
    prop, keyed by (match, question_number). Upsert (last write wins). Returns True if
    a row was written. Safe to call on every row -- it self-skips non-starter / unmapped."""
    if not _is_market_starter_prop(tier):
        return False
    qt = str(row.get("question_type", "")).strip().lower()
    line = float(row["line"]) if str(row.get("line", "")).strip() else None
    pr = price_player_prop(qt, (row.get("target_player") or None), line, game_json)
    if not pr.mapped:
        return False
    rec = _row_fields(pr, match, row.get("question_number", ""), "")
    if rec is None:
        return False
    rows = [r for r in _read(path)
            if (r.get("match"), r.get("question_number")) != (rec["match"], rec["question_number"])]
    rows.append(rec)
    _write(path, rows)
    return True


def _to_y(result) -> int | None:
    s = str(result).strip().lower()
    return 1 if s in ("1", "yes", "y", "true") else (0 if s in ("0", "no", "n", "false") else None)


def harvest(log_path: Path = LOG_PATH, pending_path: Path = PENDING_CSV,
            shadow_path: Path = SHADOW_CSV) -> int:
    """At REVIEW: join pending snapshots to resolved outcomes in the measurement log
    and append finalized rows to the shadow. Idempotent: a (match,question_number)
    already in the shadow is skipped. Returns the number of newly harvested rows."""
    pend = _read(pending_path)
    if not pend:
        return 0
    outcomes = {}
    for r in _read(log_path) if log_path.exists() else []:
        y = _to_y(r.get("result"))
        if y is not None:
            outcomes[(r.get("match"), r.get("question_number"))] = y
    have = {(r.get("match"), r.get("question_number")) for r in _read(shadow_path)}
    new = []
    for r in pend:
        k = (r.get("match"), r.get("question_number"))
        if k in have or k not in outcomes:
            continue
        r = dict(r); r["y"] = outcomes[k]
        new.append(r)
    if new:
        _write(shadow_path, _read(shadow_path) + new)
    return len(new)


def _brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ys) if ys else float("nan")


def evaluate(shadow_path: Path = SHADOW_CSV) -> None:
    rows = [r for r in _read(shadow_path) if _to_y(r.get("y")) is not None]
    if not rows:
        print(f"no resolved shadow rows yet ({shadow_path}). Auto-fills via capture_pending (lock) + harvest (review).")
        return
    y = [int(r["y"]) for r in rows]
    n = len(rows)
    tiered = _brier([float(r["tiered_p"]) for r in rows], y)
    print(f"resolved props: {n}\n  TIERED Brier = {tiered:.4f}  (the production logic)")
    print("  flat-candidate Briers (the relapse we must beat):")
    wins = True
    for c in SHADOW_OVERROUND_CANDIDATES:
        b = _brier([float(r[f"p@{c}"]) for r in rows], y)
        if tiered > b + 1e-12:
            wins = False
        print(f"    overround {c}: {b:.4f}{'' if tiered <= b + 1e-12 else '  <-- flat beat tiered (investigate)'}")
    print(f"  -> tiered beats EVERY flat constant? {'YES' if wins else 'NO'} (the right question)")
    print("\n  provenance / selection-bias audit (shipped vs realized, by source x band):")
    for src in ("exact_two_sided", "same_slate_market_prior", "global_player_prop_prior"):
        sub = [r for r in rows if r.get("overround_source") == src]
        if not sub:
            continue
        hit = sum(int(r["y"]) for r in sub) / len(sub)
        ship = sum(float(r["tiered_p"]) for r in sub) / len(sub)
        print(f"    {src:26} n={len(sub):3d}  ship {ship:.3f} vs hit {hit:.3f} ({ship-hit:+.3f})")
    print("\n  NOTE: tests TIERED-vs-flat + audits tier-b bias -- never refit an overround to this.")


def measured_global_overround(raw_dir: str = "data/raw"):
    """Re-measure the global player-prop overround from ALL two-sided quotes in the raw
    files -- the refreshable basis for GLOBAL_PLAYER_PROP_OVERROUND. Returns
    (median, n, by_band). The constant is selection-biased toward the liquid/favorite
    subset books quote two-sided; as more player types accumulate the measured median
    should rise toward the true population margin. Run periodically and update the
    constant DELIBERATELY if it drifts -- never auto-mutate (that's untracked tuning)."""
    import glob
    import json
    from .odds import odds_to_prob
    books = []
    for f in glob.glob(f"{raw_dir}/*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if isinstance(d, dict) and "data" in d:
            d = d["data"]
        for e in (d if isinstance(d, list) else [d]):
            if not isinstance(e, dict):
                continue
            for bk in e.get("bookmakers", []) or []:
                for m in bk.get("markets", []):
                    if not str(m.get("key", "")).startswith("player"):
                        continue
                    per: dict = {}
                    for o in m.get("outcomes", []):
                        nm = o.get("description") or o.get("name") or ""
                        side = (o.get("name") or "").lower()
                        try:
                            ip = float(odds_to_prob([int(o["price"])])[0])
                        except (KeyError, ValueError, TypeError):
                            continue
                        key = (nm, o.get("point"))
                        slot = "aff" if side in ("over", "yes") else ("neg" if side in ("under", "no") else "x")
                        per.setdefault(key, {})[slot] = ip
                    for d2 in per.values():
                        if "aff" in d2 and "neg" in d2 and (d2["aff"] + d2["neg"]) > 0:
                            books.append((d2["aff"] + d2["neg"], d2["aff"] / (d2["aff"] + d2["neg"])))
    if not books:
        return None, 0, {}
    bs = sorted(b for b, _ in books)
    by = {band: round(sorted([b for b, p in books if prop_band(p) == band])[
                          len([b for b, p in books if prop_band(p) == band]) // 2], 4)
          for band in ("longshot", "mid", "favorite")
          if any(prop_band(p) == band for _, p in books)}
    return round(bs[len(bs) // 2], 4), len(books), by


if __name__ == "__main__":
    h = harvest()
    print(f"harvested {h} newly-resolved prop(s) into the shadow.\n")
    evaluate()
    # REFRESH CHECK (watch item: don't let the global prior freeze at 1.045)
    from .player_prop_pricing import GLOBAL_PLAYER_PROP_OVERROUND
    med, n, by = measured_global_overround()
    if med is not None:
        print(f"\nglobal-prior refresh check: frozen={GLOBAL_PLAYER_PROP_OVERROUND}  "
              f"measured-now={med} (n={n} two-sided quotes)  by band={by}")
        print("  the two-sided subset is selection-biased (books quote both sides mainly on liquid")
        print("  players), so re-measure as coverage broadens and update GLOBAL_PLAYER_PROP_OVERROUND")
        print("  DELIBERATELY if the population median drifts (bands are ~flat in the current subset).")
