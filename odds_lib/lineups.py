"""Per-match lineup data loader for player-prop questions.

File format
-----------
``data/lineups/{YYYY-MM-DD}_{match_slug}.json``::

    {
      "match": "New Zealand vs Egypt",
      "kickoff_utc": "2026-06-22T01:00:00Z",
      "captured_at": "2026-06-22T00:35:00Z",
      "source": "manual",
      "notes": "official team sheet, posted 25min pre-kickoff",
      "players": {
        "Ben Waine":         {"team": "New Zealand", "status": "bench",   "role": "central_attacker"},
        "Chris Wood":        {"team": "New Zealand", "status": "starter", "role": "central_attacker"},
        "Mahmoud Trezeguet": {"team": "Egypt",       "status": "bench",   "role": "wide_attacker"}
      }
    }

``status`` ∈ {starter, bench_high_usage, bench_low_usage, bench_unknown,
              out_of_squad, unknown}.
``role``   ∈ {central_attacker, wide_attacker, attacking_midfielder,
              central_midfielder, defender, goalkeeper, unknown}.
``expected_minutes`` is optional (``null`` allowed) — never fabricate it.

Status discipline (important):
  - Use ``out_of_squad`` ONLY when the bench / matchday squad is confirmed
    and the player is absent from it.
  - If the starting XI is known but the bench is not, a player who is not in
    the XI is ``bench_unknown`` (they could be an unused sub) — NOT
    ``out_of_squad``. Guessing absence is a data-quality error.
  - Legacy values ``bench``/``out`` from older files are normalised to
    ``bench_unknown``/``out_of_squad`` for backward compatibility.

You only need to list the players SportsPredict is going to ask about — the
loader returns ``unknown`` for anyone not in the file. That keeps lineup
maintenance to ~2 players per match instead of all 22.

These are FEATURES only. Lineup status/role must never be turned into a
hard-coded ``p_truth`` probability — see ``odds_lib/player_features.py`` and
the model-discipline notes in ``odds_lib/decision_engine.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


LINEUP_DIR = Path("data/lineups")


VALID_STATUSES = {
    "starter",
    "bench_high_usage",
    "bench_low_usage",
    "bench_unknown",
    "out_of_squad",
    "unknown",
}

# Older lineup files used a coarser vocabulary. Map them forward so existing
# fixtures keep working without silently degrading to ``unknown``.
LEGACY_STATUS_ALIASES = {
    "bench": "bench_unknown",
    "out": "out_of_squad",
    "not_starting": "bench_unknown",
}

# Statuses that mean "in the XI" vs "not in the XI but maybe available" vs
# "confirmed absent". Used by feature extraction; kept here so the taxonomy
# lives in one place.
STARTER_STATUSES = {"starter"}
BENCH_STATUSES = {"bench_high_usage", "bench_low_usage", "bench_unknown"}
OUT_STATUSES = {"out_of_squad"}

VALID_ROLES = {
    "central_attacker", "wide_attacker", "attacking_midfielder",
    "central_midfielder", "defender", "goalkeeper", "unknown",
}


@dataclass
class PlayerContext:
    status: str = "unknown"
    role: str = "unknown"
    team: str | None = None
    expected_minutes: float | None = None
    source: str = ""


@dataclass
class MatchLineup:
    match: str
    kickoff_utc: str | None = None
    captured_at: str | None = None
    source: str = ""
    notes: str = ""
    players: dict[str, PlayerContext] = field(default_factory=dict)

    def player(self, name: str | None) -> PlayerContext:
        """Look up a player by name. Case-insensitive substring match falls
        back to last-name match. Returns unknown if no hit."""
        if not name:
            return PlayerContext()
        target = _normalize(name)
        # Exact normalized match first.
        for k, ctx in self.players.items():
            if _normalize(k) == target:
                return ctx
        # Then substring (handles "Darwin Núñez" vs "Núñez").
        for k, ctx in self.players.items():
            kn = _normalize(k)
            if target in kn or kn in target:
                return ctx
        # Then last-token match (handles "Trezeguet" vs "Mahmoud Trezeguet").
        target_tokens = target.split()
        if target_tokens:
            last = target_tokens[-1]
            for k, ctx in self.players.items():
                kn_tokens = _normalize(k).split()
                if kn_tokens and kn_tokens[-1] == last:
                    return ctx
        return PlayerContext()


# Stripped, lowercased, accent-folded for tolerant matching.
def _normalize(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.strip().lower())


def _slug(match_name: str) -> str:
    s = _normalize(match_name).replace(" vs ", "_vs_")
    return re.sub(r"[^a-z0-9_]+", "_", s)


def load_lineup(
    match_name: str,
    lineup_dir: Path = LINEUP_DIR,
    game_date: str | None = None,
) -> MatchLineup | None:
    """Find and load the lineup file for ``match_name``.

    Search strategy:
      1. If ``game_date`` given, try ``{date}_{slug}.json`` exact.
      2. Glob any file in ``lineup_dir`` whose slug substring matches.
      3. Glob any file whose stored ``match`` field equals the query.

    Returns None if no file is found. Returns a MatchLineup even if the
    file is malformed (so the caller can log + continue).
    """
    if not lineup_dir.exists():
        return None
    target_slug = _slug(match_name)
    candidates: list[Path] = []
    if game_date:
        exact = lineup_dir / f"{game_date}_{target_slug}.json"
        if exact.exists():
            candidates.append(exact)
    for p in sorted(lineup_dir.glob("*.json")):
        if target_slug in p.name and p not in candidates:
            candidates.append(p)
    if not candidates:
        # Last-resort scan: read every file and compare the "match" field.
        for p in sorted(lineup_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                if _normalize(data.get("match", "")) == _normalize(match_name):
                    candidates.append(p)
            except Exception:
                continue
    if not candidates:
        return None

    p = candidates[0]
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None

    players: dict[str, PlayerContext] = {}
    for name, info in (data.get("players") or {}).items():
        status = str(info.get("status", "unknown")).strip().lower()
        status = LEGACY_STATUS_ALIASES.get(status, status)
        role = str(info.get("role", "unknown")).strip().lower()
        if status not in VALID_STATUSES:
            status = "unknown"
        if role not in VALID_ROLES:
            role = "unknown"
        em = info.get("expected_minutes")
        try:
            expected_minutes = float(em) if em is not None else None
        except (TypeError, ValueError):
            expected_minutes = None
        players[name] = PlayerContext(
            status=status,
            role=role,
            team=info.get("team") or None,
            expected_minutes=expected_minutes,
            source=str(info.get("source", data.get("source", "")) or ""),
        )

    return MatchLineup(
        match=data.get("match", match_name),
        kickoff_utc=data.get("kickoff_utc"),
        captured_at=data.get("captured_at"),
        source=str(data.get("source", "")),
        notes=str(data.get("notes", "")),
        players=players,
    )


__all__ = [
    "PlayerContext",
    "MatchLineup",
    "LINEUP_DIR",
    "VALID_STATUSES",
    "VALID_ROLES",
    "LEGACY_STATUS_ALIASES",
    "STARTER_STATUSES",
    "BENCH_STATUSES",
    "OUT_STATUSES",
    "load_lineup",
]
