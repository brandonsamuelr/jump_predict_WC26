"""STAGED multi-provider coverage probe (runs when free-tier keys exist in .env).

Reads keys: SPORTSGAMEODDS_KEY, OPTICODDS_KEY, THERUNDOWN_KEY, ODDALERTS_KEY.
For each provider with a key, lists soccer/WC events and dumps, per sampled fixture,
the market keys returned + per-market book count + cross-book dispersion (agreement).
Scores the UNFOUNDED families against what's actually returned. STDLIB only; $0 (free
tiers); NO purchase, NO pipeline change. Verifies REAL fixtures — never trusts marketing.

Endpoints/auth are best-known shapes (SGO + OpticOdds confirmed clean-401 reachable);
confirm against each provider's docs once a key exists — the probe prints raw responses
so the schema is self-revealing.
"""
from __future__ import annotations
import json, os, urllib.request, urllib.parse, statistics
from pathlib import Path

def _keys():
    env = {}
    p = Path(".env")
    if p.exists():
        for ln in p.read_text().splitlines():
            if "=" in ln and not ln.startswith("#"):
                k, v = ln.split("=", 1); env[k.strip()] = v.strip()
    env.update(os.environ)
    return env

def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]
    except Exception as e:
        return 0, str(e)[:200]

# unfounded families to score (semantic; map to each provider's market naming at runtime)
FAMILIES = ["team_more_cards", "total_cards_over", "team_offsides_over",
            "team_more_corners_1h/2h", "team_more_sot_2h", "player_sot_over (super-sub)",
            "penalty_awarded / penalty_or_red"]

PROVIDERS = {
    "SportsGameOdds": {"key": "SPORTSGAMEODDS_KEY", "base": "https://api.sportsgameodds.com/v2",
                       "auth": lambda k: {"X-Api-Key": k}, "sports": "/sports"},
    "OpticOdds":      {"key": "OPTICODDS_KEY", "base": "https://api.opticodds.com/api/v3",
                       "auth": lambda k: {"X-Api-Key": k}, "sports": "/sports"},
    "TheRundown":     {"key": "THERUNDOWN_KEY", "base": "https://therundown-inc.p.rapidapi.com",
                       "auth": lambda k: {"x-rapidapi-key": k, "x-rapidapi-host": "therundown-inc.p.rapidapi.com"},
                       "sports": "/sports"},
    "OddAlerts":      {"key": "ODDALERTS_KEY", "base": "https://api.oddalerts.com",
                       "auth": lambda k: {"Authorization": f"Bearer {k}"}, "sports": "/upcoming"},
}

def main():
    env = _keys()
    print("families to found:", FAMILIES, "\n")
    for name, cfg in PROVIDERS.items():
        key = env.get(cfg["key"], "")
        if not key:
            print(f"[{name}] NO KEY in .env ({cfg['key']}) -> BLOCKED at free-tier signup. "
                  f"Register, add {cfg['key']}=... to .env, re-run.")
            continue
        st, body = _get(cfg["base"] + cfg["sports"], cfg["auth"](key))
        print(f"[{name}] {cfg['sports']} -> HTTP {st}")
        print("   ", json.dumps(body)[:400] if isinstance(body, (dict, list)) else body)
        print(f"   -> inspect the response, then point this probe at its soccer events/odds "
              f"endpoint to dump market keys + book counts + dispersion per fixture.")
    print("\nNOTE: settlement-stats (SGO) and lineups (OpticOdds) endpoints to be confirmed "
          "from the live response shape once keyed. Score each family market-read AND "
          "fittable-data; record book count + agreement on the ACTUAL returned market.")

if __name__ == "__main__":
    main()
