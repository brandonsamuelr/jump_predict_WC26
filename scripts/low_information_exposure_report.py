"""Task 2 — low-information exposure report (FLAGS ONLY; corrects nothing).

Flags low-information, potentially dangerous pre-lock rows for MANUAL review —
the Portugal-Q5 mechanism (a no-model row silently submitting a stale c_hat that
becomes the biggest unintended position). It NEVER changes a submission.

Stake proxy: a shadow row's realized stake materializes from (c_hat - realized
crowd), which is unknown pre-lock. So we estimate it from Task 1's per-type
historical |c_hat - crowd| (mean_abs_error): estimated_at_stake = 200 * that MAE.
This auto-recalibrates as data grows and correctly fires on the high-error SOT
family while leaving reliable shadow types (penalty/offsides/corners) alone.

    python scripts/low_information_exposure_report.py --match "Portugal vs Uzbekistan"
    python scripts/low_information_exposure_report.py --sheet <submit_or_refresh.csv>

PROVISIONAL thresholds (hand-set review-triggers, NOT validated bounds).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

import pandas as pd

from odds_lib.measurement import LOG_PATH
from odds_lib.lineups import load_lineup, BENCH_STATUSES, STARTER_STATUSES
from crowd_reliability_report import prep as crowd_prep, reliability_table

# --- PROVISIONAL flag thresholds (review-triggers, recalibrate from Task 1) ---
CAP_SOT2H_NOMODEL = 0.40        # no-model player_sot_2h_over above this -> flag
CAP_SOT2H_UNDERDOG = 0.325      # no-model UNDERDOG player_sot_2h_over above this -> flag
CAP_GOAL_BENCH = 0.12           # benched player_goal above this -> flag
CAP_ANY_PROP_NOMODEL = 0.45     # any no-model player prop above this -> flag
STAKE_FLAG = 15.0               # estimated at_stake >= this on a low-info row -> flag
ANCHOR_UNRELIABLE_MAE = 0.10    # type MAE >= this => c_hat anchor unreliable for the row
SOT_ANCHOR_PULL_MIN = 0.07      # |submit - p_hat| toward c_hat >= this (model-bearing rows)

# no-model / thin types that should fall to shadow (and are dangerous if not)
THIN_NOMODEL_TYPES = {
    "player_sot_2h_over", "player_sot_over", "player_goal", "player_goal_or_assist",
    "team_offsides_over", "team_more_fouls", "team_more_cards",
    "total_corners_over", "team_corners_over", "penalty_or_red_card",
}
PLAYER_PROP_PREFIX = "player_"


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def expected_crowd_err_map():
    """Per-type (then bucket, then overall) historical |c_hat - crowd| from Task 1."""
    log = pd.read_csv(LOG_PATH, dtype=str)
    cp = crowd_prep(log)
    if cp.empty:
        return {}, {}, 0.083
    qt = reliability_table(cp, "question_type")["mean_abs_error"].to_dict()
    bk = reliability_table(cp, "bucket")["mean_abs_error"].to_dict()
    overall = float(qt.get("ALL", 0.083))
    return qt, bk, overall


def _bucket(tier):
    t = (tier or "").upper()
    return ("MARKET" if t == "MARKET" else "ENGINE" if t == "ENGINE_GOALS"
            else "RATE_SOT" if t in ("RATE_SOT", "RATE_SOT_CMP")
            else "PROP" if t in ("PROP_OK", "PROP_THIN") else "PENDING/SHADOW")


def is_low_info(qt, tier, cls, mode, p_hat) -> bool:
    return (str(tier).upper() == "PENDING" or str(mode).lower() == "shadow"
            or str(cls).upper() == "SHADOW" or p_hat is None
            or (qt.startswith(PLAYER_PROP_PREFIX) and str(tier).upper() != "PROP_OK")
            or qt in THIN_NOMODEL_TYPES)


def flag_row(*, question_type, tier, cls, mode, c_hat, p_hat, submitted_prob,
             underdog=False, lineup_status=None, expected_err=None):
    """Return (reason_flags, estimated_at_stake) for one row. Pure / testable."""
    qt = str(question_type)
    ph = _num(p_hat)
    sub = _num(submitted_prob)
    ch = _num(c_hat)
    flags: list[str] = []
    low = is_low_info(qt, tier, cls, mode, ph)
    is_bench = lineup_status in BENCH_STATUSES or lineup_status == "out_of_squad"
    is_player = qt.startswith(PLAYER_PROP_PREFIX)

    if low:
        if ph is None:
            flags.append("NO_MODEL")
        if str(tier).upper() == "PENDING" or str(mode).lower() == "shadow" or str(cls).upper() == "SHADOW":
            flags.append("PENDING_OR_SHADOW")
        if is_player and str(tier).upper() != "PROP_OK":
            # a CONFIRMED starter on a thin/illiquid market is PROP_thin — that's
            # a thin-market note, NOT an unconfirmed-lineup risk. Only tag
            # UNCONFIRMED when the player isn't a confirmed starter.
            if lineup_status in STARTER_STATUSES:
                flags.append("PLAYER_PROP_THIN_MARKET")
            else:
                flags.append("PLAYER_PROP_UNCONFIRMED")
        if is_bench:
            flags.append("BENCH_PLAYER")
        if qt in THIN_NOMODEL_TYPES and ("PENDING_OR_SHADOW" in flags):
            flags.append("UNSUPPORTED_TYPE_FELL_TO_SHADOW")

    # sanity caps (apply to the low-info / no-model prop rows)
    if sub is not None:
        if qt == "player_sot_2h_over" and ph is None:
            cap = CAP_SOT2H_UNDERDOG if underdog else CAP_SOT2H_NOMODEL
            if sub > cap:
                flags.append("SUBMIT_ABOVE_SANITY_CAP")
            if underdog and sub > CAP_SOT2H_UNDERDOG:
                flags.append("UNDERDOG_PLAYER_2H_SOT_HIGH")
        elif qt == "player_goal" and is_bench and sub > CAP_GOAL_BENCH:
            flags.append("SUBMIT_ABOVE_SANITY_CAP")
        elif is_player and ph is None and sub > CAP_ANY_PROP_NOMODEL:
            flags.append("SUBMIT_ABOVE_SANITY_CAP")

    # stake proxy from Task 1 reliability (only meaningful for low-info rows)
    est_stake = None
    if low and expected_err is not None:
        est_stake = round(200.0 * expected_err, 1)
        if est_stake >= STAKE_FLAG:
            flags.append("HIGH_STAKE_LOW_INFO")
        if expected_err >= ANCHOR_UNRELIABLE_MAE:
            flags.append("LARGE_CROWD_EXPOSURE_PROXY")

    # SOT anchor-pull: model-bearing SOT row dragged toward stale c_hat (the Q9 case)
    if str(tier).upper() in ("RATE_SOT", "RATE_SOT_CMP") and ph is not None and sub is not None and ch is not None:
        moved = sub - ph
        toward_chat = (ch - ph)
        if abs(moved) >= SOT_ANCHOR_PULL_MIN and moved * toward_chat > 0:
            flags.append("SOT_ANCHOR_PULL")

    return list(dict.fromkeys(flags)), est_stake   # dedupe, keep order


_SEVERITY = {"HIGH_STAKE_LOW_INFO": 5, "SUBMIT_ABOVE_SANITY_CAP": 4,
             "UNDERDOG_PLAYER_2H_SOT_HIGH": 4, "SOT_ANCHOR_PULL": 3,
             "BENCH_PLAYER": 3, "UNSUPPORTED_TYPE_FELL_TO_SHADOW": 2,
             "LARGE_CROWD_EXPOSURE_PROXY": 2, "PLAYER_PROP_UNCONFIRMED": 1,
             "PLAYER_PROP_THIN_MARKET": 1, "NO_MODEL": 0, "PENDING_OR_SHADOW": 0}


def _favorite_underdog(sheet_match: pd.DataFrame, match: str):
    """Infer (favorite, underdog) from the team_win row's submitted prob, if present."""
    tw = sheet_match[sheet_match["type"] == "team_win"]
    teams = [t.strip() for t in str(match).split(" vs ")]
    if tw.empty or len(teams) != 2:
        return None, None
    r = tw.iloc[0]
    fav = r.get("_target_team") or teams[0]
    sub = _num(r.get("SUBMIT"))
    if sub is None:
        return None, None
    other = teams[1] if fav == teams[0] else teams[0]
    return (fav, other) if sub >= 0.5 else (other, fav)


def _load_sheet(args):
    if args.sheet:
        return pd.read_csv(args.sheet, dtype=str).fillna("")
    sd = Path("data/submission_sheets")
    if args.match:
        slug = args.match.lower().replace(" ", "_").replace("vs", "v")
        hits = sorted(sd.glob(f"*_{slug}_refresh.csv"))
        if hits:
            return pd.read_csv(hits[-1], dtype=str).fillna("")
    # fallback: newest optimized submit sheet
    opt = sorted(sd.glob("*optimized_submit_sheet.csv"))
    if not opt:
        raise SystemExit("no sheet found; pass --sheet")
    df = pd.read_csv(opt[-1], dtype=str).fillna("")
    return df[df["match"] == args.match] if args.match else df


def _questions_lookup():
    out = {}
    for f in Path("data/submission_sheets").glob("*questions*.csv"):
        try:
            q = pd.read_csv(f, dtype=str).fillna("")
            for _, r in q.iterrows():
                out[(r.get("match"), r.get("question_number"))] = r
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match")
    ap.add_argument("--sheet")
    args = ap.parse_args()

    sheet = _load_sheet(args)
    if "q" not in sheet.columns and "question_number" in sheet.columns:
        sheet = sheet.rename(columns={"question_number": "q"})
    qt_mae, bk_mae, overall_mae = expected_crowd_err_map()
    qlook = _questions_lookup()

    out = []
    for match, g in sheet.groupby("match"):
        fav, dog = _favorite_underdog(g, match)
        lineup = load_lineup(match)
        for _, r in g.iterrows():
            qt = str(r.get("type", ""))
            tier, cls, mode = r.get("tier", ""), r.get("class", ""), r.get("mode", "")
            ch, ph, sub = r.get("c_hat", ""), r.get("p_hat", ""), r.get("SUBMIT", "")
            qrow = qlook.get((match, r.get("q")), {})
            tteam = qrow.get("target_team", "") if hasattr(qrow, "get") else ""
            tplayer = qrow.get("target_player", "") if hasattr(qrow, "get") else ""
            underdog = bool(dog) and tteam == dog
            lu_status = lineup.player(tplayer).status if (lineup is not None and tplayer) else None
            exp_err = qt_mae.get(qt) or bk_mae.get(_bucket(tier)) or overall_mae
            flags, est_stake = flag_row(
                question_type=qt, tier=tier, cls=cls, mode=mode, c_hat=ch, p_hat=ph,
                submitted_prob=sub, underdog=underdog, lineup_status=lu_status,
                expected_err=exp_err)
            if not flags:
                continue
            sev = max((_SEVERITY.get(f, 0) for f in flags), default=0)
            out.append({
                "match": match, "q": r.get("q"), "type": qt,
                "team": tteam or "-", "player": (tplayer or "-")[:16],
                "tier": tier, "mode": mode, "c_hat": _num(ch),
                "p_hat": _num(ph) if _num(ph) is not None else "",
                "SUBMIT": _num(sub), "est_stake": est_stake if est_stake is not None else "",
                "lineup": lu_status or "-", "udog": "Y" if underdog else "",
                "_sev": sev, "flags": ",".join(flags),
            })

    print("=" * 110)
    print("TASK 2 — LOW-INFORMATION EXPOSURE  (FLAGS ONLY — corrects nothing; for manual review)")
    print("=" * 110)
    if not out:
        print("no flagged rows."); return
    rep = pd.DataFrame(out).sort_values("_sev", ascending=False).drop(columns=["_sev"])
    pd.set_option("display.width", 220, "display.max_colwidth", 40, "display.max_rows", 60)
    print(rep.to_string(index=False))
    print(f"\n{len(rep)} flagged rows. est_stake = 200 * (type's historical |c_hat-crowd| from Task 1).")
    print("PROVISIONAL thresholds (review-triggers, not validated). FLAGS ONLY — change no submission.")
    print("Action on a flag is MANUAL: confirm the input (lineup/line/market), or shadow it — never auto-correct.")


if __name__ == "__main__":
    main()
