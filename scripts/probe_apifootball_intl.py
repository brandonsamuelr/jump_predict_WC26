"""$0 FREE-TIER feasibility probe: API-Football international match-stat coverage.

STDLIB ONLY (urllib) — installs NOTHING, cannot touch the live .venv. Reads the key
from API_FOOTBALL_KEY (env or .env). Direct host v3.football.api-sports.io, header
x-apisports-key. Budget-guarded (hard stop well under the 100/day free cap). Emits a
scratch JSON to data/historical/intl_probe/. NO purchase, NO shipping, report only.

    API_FOOTBALL_KEY=xxxx .venv/bin/python scripts/probe_apifootball_intl.py
    # or add API_FOOTBALL_KEY=xxxx to .env then just run it

PASS bar: fields populated on visible intl matches AND paid tier plausibly >=15-20
populated intl matches/team for the WEAK teams (Haiti/SA/Panama). Populated-but-thin = FAIL.
"""
from __future__ import annotations
import json, os, sys, time, urllib.request, urllib.parse
from pathlib import Path

HOST = "https://v3.football.api-sports.io"
OUT = Path("data/historical/intl_probe/apifootball_probe.json")
BUDGET = 90          # hard stop under the 100/day free cap
PACE = 7.0           # free tier throttles ~10 req/min -> 7s spacing (~8.5/min, safe)
WEAK = ["Haiti", "South Africa", "Panama", "Jordan"]   # ugly sample (CONCACAF/CAF/AFC minnows)
CONTROLS = ["Brazil", "Mexico"]                         # must be covered or API is unusable
# free tier season window observed: 2022-2024 ("Free plans do not have access to this season")
SEASONS = [2022, 2023, 2024]
FIELDS = ["Fouls", "Corner Kicks", "Offsides", "Yellow Cards", "Red Cards",
          "Total Shots", "Shots on Goal"]
SAMPLE_FIXTURES_PER_TEAM = 3

def _key() -> str:
    k = os.environ.get("API_FOOTBALL_KEY", "").strip()
    if not k:
        envf = Path(".env")
        if envf.exists():
            for ln in envf.read_text().splitlines():
                if ln.startswith("API_FOOTBALL_KEY="):
                    k = ln.split("=", 1)[1].strip().strip('"').strip("'")
    if not k:
        sys.exit("BLOCKED: no API_FOOTBALL_KEY in env or .env. Register a free key at "
                 "dashboard.api-football.com (no card) and provide it. Probe staged, not run.")
    return k

_calls = {"n": 0, "remaining": None}
def get(path: str, **params):
    if _calls["n"] >= BUDGET:
        raise SystemExit(f"BUDGET stop at {_calls['n']} calls (free cap protection).")
    url = f"{HOST}/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"x-apisports-key": _key(),
                                               "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                _calls["n"] += 1
                _calls["remaining"] = r.headers.get("x-ratelimit-requests-remaining") or _calls["remaining"]
                body = json.loads(r.read().decode())
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:   # per-minute throttle -> back off
                print(f"  429 throttle on {path}; backing off 65s (attempt {attempt+1})")
                time.sleep(65); continue
            print(f"  HTTPError {e.code} on {path}: skipping"); return {"errors": {"http": e.code}, "response": []}
    time.sleep(PACE)
    if body.get("errors"):
        # surface auth/plan errors loudly (e.g. season not allowed on free tier)
        print(f"  API note ({path}): {body['errors']}")
    return body

def national_team_id(name: str):
    b = get("teams", search=name)
    nat = [t for t in b.get("response", []) if t.get("team", {}).get("national")]
    pick = (nat or b.get("response", []))
    if not pick:
        return None, None
    t = pick[0]["team"]
    return t["id"], t["name"]

def fixture_stats_populated(fid: int):
    b = get("fixtures/statistics", fixture=fid)
    resp = b.get("response", [])
    if not resp:
        return None
    # check the first team entry's field population (value not None)
    types = {s["type"]: s.get("value") for s in resp[0].get("statistics", [])}
    return {f: (types.get(f) is not None) for f in FIELDS}

def probe_team(name: str):
    tid, tname = national_team_id(name)
    rec = {"query": name, "team_id": tid, "team_name": tname,
           "fixtures_exposed": 0, "seasons_seen": [], "sampled": [], "populated_count": 0}
    if tid is None:
        rec["error"] = "no national team found"; return rec
    fixtures = []
    for s in SEASONS:
        b = get("fixtures", team=tid, season=s)
        n = b.get("results", 0)
        if n:
            rec["seasons_seen"].append({"season": s, "fixtures": n})
            rec["fixtures_exposed"] += n
            fixtures += b.get("response", [])
        if _calls["n"] >= BUDGET: break
    # sample a few finished fixtures, check stat population + referee presence
    fin = [f for f in fixtures if (f.get("fixture", {}).get("status", {}).get("short") == "FT")]
    for f in fin[:SAMPLE_FIXTURES_PER_TEAM]:
        if _calls["n"] >= BUDGET: break
        fx = f["fixture"]
        pop = fixture_stats_populated(fx["id"])
        rec["sampled"].append({
            "fixture_id": fx["id"],
            "match": f["teams"]["home"]["name"] + " vs " + f["teams"]["away"]["name"],
            "date": fx.get("date", "")[:10],
            "referee": fx.get("referee"),
            "fields_populated": pop,
        })
        if pop and all(pop[x] for x in ["Fouls", "Corner Kicks", "Yellow Cards", "Shots on Goal"]):
            rec["populated_count"] += 1
    return rec

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    status = get("status")
    out = {"status": status.get("response", status), "teams": {}, "fields_checked": FIELDS,
           "half_splits_available": False,  # API-Football stats are full-match only — flag
           "note": "free-tier season-limited; tests FIELD POPULATION on visible matches"}
    print("=== STATUS ===", json.dumps(out["status"], indent=2)[:800])
    for name in WEAK + CONTROLS:
        print(f"\n--- probing {name} (calls so far {_calls['n']}) ---")
        try:
            out["teams"][name] = probe_team(name)
        except SystemExit as e:
            out["teams"][name] = {"stopped": str(e)}; print(e); break
        r = out["teams"][name]
        print(f"  {name}: id={r.get('team_id')} exposed={r.get('fixtures_exposed')} "
              f"sampled={len(r.get('sampled',[]))} fully_populated={r.get('populated_count')}")
    out["calls_used"] = _calls["n"]; out["rate_remaining"] = _calls["remaining"]
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}  (calls used {_calls['n']}, remaining {_calls['remaining']})")
    # quick verdict scaffold
    print("\n=== POPULATION SUMMARY (weak teams are the test) ===")
    for name in WEAK + CONTROLS:
        r = out["teams"].get(name, {})
        tag = "WEAK" if name in WEAK else "ctrl"
        print(f"  [{tag}] {name:14s} exposed={r.get('fixtures_exposed','?'):>4} "
              f"fully_populated_sampled={r.get('populated_count','?')}/{len(r.get('sampled',[]))}")

if __name__ == "__main__":
    main()
