"""Build the historical team stat-lines corpus from football-data.co.uk.

Deliverable 1 of the Bucket-B foundation. Each football-data.co.uk match row
already carries BOTH team match stats (shots / SOT / fouls / corners / cards)
AND closing betting odds (1X2 + O/U 2.5), so there is no separate odds-join:
the market context is computed from the same row.

CRITICAL: the market-context features (team_win_prob, favorite_gap,
total_line_prob) are computed with the LIVE pipeline's own de-vig functions
(`odds_lib.odds.odds_to_prob` + `remove_vig`) so they are numerically identical
to what the live pipeline computes from the Odds API. We de-vig each book
separately then average across books, mirroring the live consensus
(`market_prob` = mean of per-book novig probs). Decimal odds are converted to
American purely as a format step before `odds_to_prob`; the de-vig itself is the
pipeline's `remove_vig` — no hand-rolled de-vig.

Output: ``data/historical/stat_lines.csv`` — one row per team per match.

    python scripts/build_stat_lines.py            # full default corpus
    python scripts/build_stat_lines.py --seasons 2223,2324,2425   # subset
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests

from odds_lib.odds import odds_to_prob, remove_vig

BASE = "https://www.football-data.co.uk/mmz4281"
CACHE = Path("data/historical/raw_footballdata")
OUT = Path("data/historical/stat_lines.csv")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Main European divisions that carry the full stat-column set.
DIVISIONS = ["E0", "E1", "E2", "E3", "SC0", "D1", "D2", "I1", "I2",
             "SP1", "SP2", "F1", "F2", "N1", "B1", "P1", "T1", "G1"]
LEAGUE_NAME = {
    "E0": "England-Premier", "E1": "England-Championship", "E2": "England-L1",
    "E3": "England-L2", "SC0": "Scotland-Prem", "D1": "Germany-Bundesliga",
    "D2": "Germany-Bundesliga2", "I1": "Italy-SerieA", "I2": "Italy-SerieB",
    "SP1": "Spain-LaLiga", "SP2": "Spain-LaLiga2", "F1": "France-Ligue1",
    "F2": "France-Ligue2", "N1": "Netherlands-Eredivisie", "B1": "Belgium-ProLeague",
    "P1": "Portugal-Primeira", "T1": "Turkey-SuperLig", "G1": "Greece-SuperLeague",
}
# 2010-11 .. 2025-26
DEFAULT_SEASONS = [f"{y:02d}{y + 1:02d}" for y in range(10, 26)]

STAT_COLS = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
BASE_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


# --- de-vig (routes through the pipeline's functions) -----------------------
def _dec_to_american(d: float) -> float:
    """Decimal -> American (format conversion only; de-vig is remove_vig)."""
    d = float(d)
    return (d - 1.0) * 100.0 if d >= 2.0 else -100.0 / (d - 1.0)


def _devig_book(decimals: list[float]) -> np.ndarray | None:
    """De-vig one book's complete market via the pipeline's functions.

    Returns None if any leg is missing/invalid (so the book is skipped).
    """
    if any(d is None or not np.isfinite(d) or d <= 1.0 for d in decimals):
        return None
    american = [_dec_to_american(d) for d in decimals]
    raw = odds_to_prob(american, "american")   # pipeline: American -> raw prob
    return remove_vig(raw)                      # pipeline: proportional de-vig


def _consensus(book_decimals: list[list[float]]) -> np.ndarray | None:
    """Mean of per-book de-vigged probs (mirrors live market_prob = mean novig)."""
    devigged = [v for v in (_devig_book(b) for b in book_decimals) if v is not None]
    if not devigged:
        return None
    return np.mean(np.vstack(devigged), axis=0)


def _col(row: pd.Series, name: str):
    v = row.get(name)
    return float(v) if (name in row.index and pd.notna(v)) else None


def _market_features(row: pd.Series) -> dict:
    """team(home)-oriented 1X2 + totals features, de-vigged & book-averaged."""
    # 1X2 from Bet365 (B365H/D/A) + Pinnacle (PSH/PSD/PSA)
    books_1x2 = []
    for h, d, a in [("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA")]:
        books_1x2.append([_col(row, h), _col(row, d), _col(row, a)])
    p_1x2 = _consensus(books_1x2)   # [pH, pD, pA] or None

    # Totals O/U 2.5 from Bet365 + Pinnacle
    books_tot = []
    for o, u in [("B365>2.5", "B365<2.5"), ("P>2.5", "P<2.5")]:
        books_tot.append([_col(row, o), _col(row, u)])
    p_tot = _consensus(books_tot)   # [p_over, p_under] or None

    out = {"home_win_prob": np.nan, "draw_prob": np.nan, "away_win_prob": np.nan,
           "total_line_prob": np.nan}
    if p_1x2 is not None:
        out["home_win_prob"], out["draw_prob"], out["away_win_prob"] = map(float, p_1x2)
    if p_tot is not None:
        out["total_line_prob"] = float(p_tot[0])  # de-vigged P(over 2.5)
    return out


# --- download ---------------------------------------------------------------
def _fetch(season: str, div: str) -> pd.DataFrame | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"{season}_{div}.csv"
    if cache_path.exists():
        raw = cache_path.read_bytes()
    else:
        url = f"{BASE}/{season}/{div}.csv"
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        except Exception as e:
            print(f"  {season}/{div}: request error {e}")
            return None
        if r.status_code != 200 or not r.content:
            print(f"  {season}/{div}: HTTP {r.status_code} (skip)")
            return None
        raw = r.content
        cache_path.write_bytes(raw)
        time.sleep(0.3)  # polite: only on new downloads
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    except Exception as e:
        print(f"  {season}/{div}: parse error {e}")
        return None
    missing = [c for c in BASE_COLS + STAT_COLS if c not in df.columns]
    if missing:
        print(f"  {season}/{div}: missing cols {missing} (skip file)")
        return None
    return df


# --- main -------------------------------------------------------------------
def build(seasons: list[str], divisions: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for season in seasons:
        for div in divisions:
            df = _fetch(season, div)
            if df is None:
                continue
            # keep only rows with full stat lines + valid result
            df = df.dropna(subset=BASE_COLS + STAT_COLS)
            kept = 0
            for _, r in df.iterrows():
                date = pd.to_datetime(r["Date"], dayfirst=True, errors="coerce")
                if pd.isna(date):
                    continue
                mf = _market_features(r)
                home, away = str(r["HomeTeam"]).strip(), str(r["AwayTeam"]).strip()
                mid = f"{season}_{div}_{date:%Y%m%d}_{home}_{away}".replace(" ", "")
                ftr = str(r["FTR"]).strip().upper()
                base = {
                    "match_id": mid, "date": date.date().isoformat(),
                    "league": LEAGUE_NAME.get(div, div), "div": div, "season": season,
                    "ftr": ftr, "home_win_prob": mf["home_win_prob"],
                    "draw_prob": mf["draw_prob"], "away_win_prob": mf["away_win_prob"],
                    "total_line_prob": mf["total_line_prob"],
                }
                # explode to one row per team
                for side in ("home", "away"):
                    is_home = side == "home"
                    pre = "H" if is_home else "A"
                    opp = "A" if is_home else "H"
                    team_wp = mf["home_win_prob"] if is_home else mf["away_win_prob"]
                    opp_wp = mf["away_win_prob"] if is_home else mf["home_win_prob"]
                    res = "win" if ftr == pre else ("draw" if ftr == "D" else "loss")
                    rows.append({
                        **base,
                        "team": home if is_home else away,
                        "opponent": away if is_home else home,
                        "is_home": int(is_home),
                        "goals_for": int(r[f"FT{pre}G"]), "goals_against": int(r[f"FT{opp}G"]),
                        "shots_for": int(r[f"{pre}S"]), "shots_against": int(r[f"{opp}S"]),
                        "sot_for": int(r[f"{pre}ST"]), "sot_against": int(r[f"{opp}ST"]),
                        "fouls_for": int(r[f"{pre}F"]), "fouls_against": int(r[f"{opp}F"]),
                        "corners_for": int(r[f"{pre}C"]), "corners_against": int(r[f"{opp}C"]),
                        "yellows_for": int(r[f"{pre}Y"]), "yellows_against": int(r[f"{opp}Y"]),
                        "reds_for": int(r[f"{pre}R"]), "reds_against": int(r[f"{opp}R"]),
                        "result_for": res,
                        # market context (pipeline-de-vigged)
                        "team_win_prob": team_wp, "opp_win_prob": opp_wp,
                        # signed per-team supremacy = this team's win prob - opp's
                        # (== +fav_underdog_gap for the favorite, - for the dog)
                        "favorite_gap": (team_wp - opp_wp)
                        if (team_wp is not None and opp_wp is not None
                            and np.isfinite(team_wp) and np.isfinite(opp_wp)) else np.nan,
                    })
                    kept += 1
            if kept:
                print(f"  {season}/{div}: {kept} team-rows")
    out = pd.DataFrame(rows)
    # |gap| = the live pipeline's exact match-level fav_underdog_gap magnitude
    out["fav_underdog_gap_abs"] = out["favorite_gap"].abs()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=",".join(DEFAULT_SEASONS))
    ap.add_argument("--divisions", default=",".join(DIVISIONS))
    args = ap.parse_args()
    seasons = args.seasons.split(",")
    divisions = args.divisions.split(",")
    print(f"building corpus: {len(seasons)} seasons x {len(divisions)} divisions")
    df = build(seasons, divisions)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # --- summary ---
    n_matches = df["match_id"].nunique()
    has_mkt = df["favorite_gap"].notna()
    has_tot = df["total_line_prob"].notna()
    print(f"\nwrote {OUT}")
    print(f"  team-rows: {len(df):,}   matches: {n_matches:,}")
    print(f"  leagues: {df['league'].nunique()}   seasons: {df['season'].nunique()} "
          f"({df['season'].min()}..{df['season'].max()})")
    print(f"  date range: {df['date'].min()} .. {df['date'].max()}")
    print(f"  rows with 1X2 market features: {has_mkt.sum():,} ({has_mkt.mean():.1%})")
    print(f"  rows with totals feature:      {has_tot.sum():,} ({has_tot.mean():.1%})")
    g = df.loc[has_mkt, "favorite_gap"]
    print("\n  favorite_gap (signed, this team - opp) distribution:")
    print(f"    min {g.min():+.3f}  p10 {g.quantile(.10):+.3f}  p25 {g.quantile(.25):+.3f}  "
          f"median {g.median():+.3f}  p75 {g.quantile(.75):+.3f}  p90 {g.quantile(.90):+.3f}  "
          f"max {g.max():+.3f}")
    print(f"    even matchups |gap|<0.10: {(df['fav_underdog_gap_abs'] < 0.10).mean():.1%}   "
          f"heavy favorites gap>0.40: {(g > 0.40).mean():.1%}")


if __name__ == "__main__":
    main()
