"""Near-KO player-SOT poll (re-runnable at each timepoint). Checks OddsPapi 6 books +
The Odds API player_shots_on_target on the imminent WC fixture. Cap-aware (1 OddsPapi
call/run). Run: .venv/bin/python scripts/poll_player_sot.py"""
import urllib.request, json, urllib.parse, datetime as dt, os
def env(k): return [l.split("=",1)[1].strip() for l in open(".env") if l.startswith(k+"=")][0]
UA="Mozilla/5.0 AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
OP=env("ODDSPAPI_KEY"); OA=env("ODDS_API_KEY")
def opp(path,**p):
    p["apiKey"]=OP; u=f"https://api.oddspapi.io/v4{path}?"+urllib.parse.urlencode(p)
    with urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":UA}),timeout=40) as r: return json.loads(r.read())
def oa(path,**p):
    p["apiKey"]=OA; u=f"https://api.the-odds-api.com/v4/{path}?"+urllib.parse.urlencode(p)
    try:
        r=urllib.request.urlopen(u,timeout=30); return json.loads(r.read())
    except urllib.error.HTTPError as e: return e.read().decode()[:160]
SHOT=set(map(str,json.load(open("/tmp/sot_ids.json")))) if os.path.exists("/tmp/sot_ids.json") else set()
now=dt.datetime.now(dt.timezone.utc)
# imminent fixture (OddsPapi)
fx=opp("/fixtures",tournamentId=16,hasOdds="true"); items=fx if isinstance(fx,list) else fx.get("data",[])
fut=sorted([( (dt.datetime.fromisoformat(f["startTime"].replace("Z","+00:00"))-now).total_seconds()/60, f) for f in items if f.get("startTime")], key=lambda x:x[0])
fut=[x for x in fut if x[0]>-15]
mins,f=fut[0]; FID=f["fixtureId"]
print(f"=== POLL @ T{mins:+.0f}min  fixture {FID} ===")
od=opp("/odds",fixtureId=FID,bookmakers="pinnacle,bet365,fanduel,draftkings,1xbet,betano",oddsFormat="decimal",verbosity="3")
print(f"OddsPapi {od.get('participant1Name')} v {od.get('participant2Name')}:")
for bk,bd in (od.get("bookmakerOdds") or {}).items():
    M=bd.get("markets",{}) or {}; sot=[m for m in M if m in SHOT]
    npl=max([len([pk for o in (M[m].get('outcomes',{}) or {}).values() for pk in (o.get('players',{}) or {}) if pk not in ('0',0)]) for m in sot]+[0])
    print(f"  [{bk}] player-SOT markets={len(sot)} real-players={npl}")
# Odds API same fixture
ev=oa("sports/soccer_fifa_world_cup/events")
eg=[e for e in ev if "Ecuador" in e.get("home_team","")+e.get("away_team","") and "Germany" in e.get("home_team","")+e.get("away_team","")] if isinstance(ev,list) else []
if eg:
    o2=oa(f"sports/soccer_fifa_world_cup/events/{eg[0]['id']}/odds",regions="us,uk,eu",markets="player_shots_on_target",oddsFormat="decimal")
    if isinstance(o2,dict):
        bks={b["title"]:len(m.get("outcomes",[])) for b in o2.get("bookmakers",[]) for m in b.get("markets",[]) if m["key"]=="player_shots_on_target"}
        print(f"Odds API player_shots_on_target books: {bks}")
