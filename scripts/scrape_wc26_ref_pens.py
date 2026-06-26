"""BOUNDED Transfermarkt.us scrape: WC26 center-ref CAREER penalty/match aggregates.

Uses requests + bs4 (already present in the env — installs NOTHING). transfermarkt.us
is reachable from here with no Cloudflare challenge, so no headless browser needed.
Per ref: 1 search (-> id) + 1 career aggregate page (saison_id=ges). Polite delay.
Career page table cols: [_, Competition, Appearances, Yellow, Yellow-Red, Red, Penalties].
Emits data/historical/intl_referees/wc26_ref_career.csv. No odds_lib, no shipping.
"""

from __future__ import annotations

import csv, re, time, sys
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE = "https://www.transfermarkt.us"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
OUT = Path("data/historical/intl_referees/wc26_ref_career.csv")

REFS = [  # 51 WC26 center referees (Wikipedia 2026 WC officials)
 "Omar Al Ali","Abdulrahman Al-Jassim","Khalid Al-Turais","Alireza Faghani","Ma Ning",
 "Adham Makhadmeh","Ilgiz Tantashev","Yusuke Araki","Pierre Atcho","Dahane Beida",
 "Mustapha Ghorbal","Jalal Jayed","Amin Omar","Abongile Tom","Iván Barton",
 "Juan Gabriel Calderón","Ismail Elfath","Oshane Nation","Drew Fischer","Katia Itzel García",
 "Saíd Martínez","Tori Penso","César Arturo Ramos","Ramon Abatti","Juan Gabriel Benítez",
 "Raphael Claus","Yael Falcón","Cristián Garay","Darío Herrera","Kevin Ortega",
 "Andrés Rojas","Wilton Sampaio","Gustavo Tejera","Facundo Tello","Jesús Valenzuela",
 "Campbell-Kirk Kawana-Waugh","Espen Eskås","Alejandro Hernández Hernández","István Kovács",
 "François Letexier","Danny Makkelie","Szymon Marciniak","Maurizio Mariani","Glenn Nyberg",
 "Michael Oliver","João Pinheiro","Sandro Schärer","Anthony Taylor","Clément Turpin",
 "Slavko Vinčić","Felix Zwayer"]

INTL = ("world cup","european championship","nations league","copa am","gold cup",
        "africa cup","afcon","asian cup","confederations","olympic","friendl")

def _num(s):
    s=(s or "").replace(",","").replace(".","").strip()
    return int(s) if s.isdigit() else 0

def find_id(name):
    r=requests.get(BASE+"/schnellsuche/ergebnis/schnellsuche",params={"query":name},headers=UA,timeout=25)
    m=re.search(r'/[^"\']+/profil/schiedsrichter/(\d+)',r.text)
    if not m: return None
    return re.search(r'(/[^"\']+/profil/schiedsrichter/\d+)',r.text).group(1)

def career(prof_path):
    url=BASE+prof_path.split("/profil/")[0]+f"/profil/schiedsrichter/{prof_path.rstrip('/').split('/')[-1]}/plus/0?funktion=1&saison_id=ges"
    s=BeautifulSoup(requests.get(url,headers=UA,timeout=25).text,"html.parser")
    t=s.find("table",class_="items")
    if not t: return None
    rows=[]
    for tr in t.find_all("tr"):
        c=[td.get_text(strip=True) for td in tr.find_all(["th","td"])]
        if len(c)>=7: rows.append(c)
    # rows[1] (first body row, blank competition) = career TOTAL
    body=[r for r in rows if r[0]=="" or r[1]!="Competition"]
    body=[r for r in rows[1:]]
    tot=body[0]  # total row (blank competition name)
    tot_m,tot_pen=_num(tot[2]),_num(tot[6])
    intl_m=intl_pen=0
    for r in body[1:]:
        comp=r[1].lower()
        if any(k in comp for k in INTL):
            intl_m+=_num(r[2]); intl_pen+=_num(r[6])
    return tot_m,tot_pen,intl_m,intl_pen

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    out=[]; ok=fail=0
    for nm in REFS:
        try:
            pid=find_id(nm); time.sleep(0.5)
            if not pid: print(f"  FAIL(no id) {nm}"); fail+=1; continue
            res=career(pid); time.sleep(0.5)
            if not res: print(f"  FAIL(no table) {nm}"); fail+=1; continue
            tm,tp,im,ip=res
            out.append({"ref":nm,"career_matches":tm,"career_pens":tp,
                        "career_pen_rate":round(tp/tm,4) if tm else 0,
                        "intl_matches":im,"intl_pens":ip,
                        "intl_pen_rate":round(ip/im,4) if im else ""})
            ok+=1
            print(f"  {nm:34s} matches={tm:4d} pens={tp:4d} rate={tp/tm:.3f}  intl_m={im} intl_pen={ip}" if tm else f"  {nm}: 0 matches")
        except Exception as e:
            print(f"  ERR {nm}: {type(e).__name__} {str(e)[:80]}"); fail+=1
    with open(OUT,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["ref","career_matches","career_pens","career_pen_rate","intl_matches","intl_pens","intl_pen_rate"])
        w.writeheader(); w.writerows(out)
    print(f"\nscraped {ok}/{len(REFS)} ok, {fail} failed -> {OUT}")

if __name__=="__main__":
    main()
