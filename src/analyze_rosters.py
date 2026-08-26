"""
analyze_rosters.py — classifica squadre per fantapunti attesi + formazione tipo della mia squadra.

Legge il file rose finali (foglio 'ROSE': gruppi di colonne team/costo), abbina ogni giocatore
alle proiezioni (data/processed/player_values.csv), calcola per ogni squadra il MIGLIOR XI
provando tutti i moduli Mantra ufficiali (assegnazione greedy per slot), e produce:
  - classifica 1°-12° per fantapunti attesi dell'XI titolare (+ profondità rosa)
  - per la MIA squadra: modulo migliore, XI titolare, panchina, note (infortuni/ballottaggi).

Uso: python src/analyze_rosters.py [percorso_file_rose.xlsx]
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

try:
    from . import config as C
except ImportError:
    import config as C

try:
    from rapidfuzz import fuzz
    def ratio(a, b): return fuzz.ratio(a, b)
except Exception:
    def ratio(a, b):
        a, b = set(a), set(b); return 100*len(a & b)/max(1, len(a | b))

ROSTERS_FILE = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(C.RAW_DIR, "isacufantafebio-rosters-1787236570161.xlsx")


def parse_rosters(path):
    raw = pd.read_excel(path, sheet_name="ROSE", header=None)
    teams = {}
    c = 0
    ncols = raw.shape[1]
    while c < ncols:
        name = raw.iat[0, c]
        if isinstance(name, str) and name.strip() and str(raw.iat[0, c+1]).strip().lower() == "costo":
            players = []
            for r in range(1, raw.shape[0]):
                pn = raw.iat[r, c]
                if isinstance(pn, str) and pn.strip() and pn.strip().lower() != "totale":
                    cost = raw.iat[r, c+1]
                    cost = int(cost) if pd.notna(cost) else 0
                    players.append((pn.strip(), cost))
            teams[name.strip()] = players
            c += 3   # team, costo, spacer
        else:
            c += 1
    return teams


def match_players(teams, pv):
    names = pv["Nome"].astype(str).tolist()
    idx_by_lower = {n.lower(): i for i, n in enumerate(names)}
    rows = []
    unmatched = []
    for team, plist in teams.items():
        for pn, cost in plist:
            key = pn.lower()
            mi = idx_by_lower.get(key)
            if mi is None:
                best, bs = None, -1
                for i, n in enumerate(names):
                    s = ratio(key, n.lower())
                    if s > bs:
                        bs, best = s, i
                mi = best if bs >= 85 else None
                if mi is None:
                    unmatched.append((team, pn)); continue
            r = pv.iloc[mi]
            roles = [x for x in str(r["mantra_roles"] or "").split(";") if x] or [str(r["role"])]
            rows.append({"team": team, "roster_name": pn, "cost": cost, "Id": int(r["Id"]),
                         "Nome": r["Nome"], "roles": roles, "macro": r["macro"],
                         "exp_total": float(r["exp_total"] or 0), "exp_pv": float(r["exp_pv"] or 0),
                         "exp_fm": float(r["exp_fm"] or 0), "VAR": float(r["VAR"] or 0),
                         "games_out": int(r.get("games_out") or 0),
                         "status": str(r.get("starter_status") or ""),
                         "pen": int(r.get("pen_rank") or 0)})
    return pd.DataFrame(rows), unmatched


def best_lineup(players, module):
    """Greedy: assign players to the module's 11 slots maximizing summed exp_total."""
    slots = [set(s) for s in C.MANTRA_MODULES[module]]
    slots = [{"i": i, "roles": s, "p": None} for i, s in enumerate(slots)]
    pool = sorted(players, key=lambda p: -p["exp_total"])
    used = set()
    order = sorted(range(len(slots)),
                   key=lambda k: (len(slots[k]["roles"]),
                                  sum(1 for p in pool if p["roles"] and set(p["roles"]) & slots[k]["roles"])))
    for k in order:
        s = slots[k]
        for p in pool:
            if p["Id"] in used:
                continue
            if set(p["roles"]) & s["roles"]:
                s["p"] = p; used.add(p["Id"]); break
    val = sum(s["p"]["exp_total"] for s in slots if s["p"])
    filled = sum(1 for s in slots if s["p"])
    return {"module": module, "slots": sorted(slots, key=lambda x: x["i"]),
            "val": val, "filled": filled}


def best_module(players):
    best = None
    for m in C.MANTRA_MODULES:
        r = best_lineup(players, m)
        sc = r["filled"]*1e6 + r["val"]
        if best is None or sc > best["sc"]:
            best = {**r, "sc": sc}
    return best


def run():
    pv = pd.read_csv(os.path.join(C.PROCESSED_DIR, "player_values.csv"))
    teams = parse_rosters(ROSTERS_FILE)
    df, unmatched = match_players(teams, pv)

    results = []
    per_team_players = {}
    for team in teams:
        tp = df[df["team"] == team].to_dict("records")
        per_team_players[team] = tp
        bm = best_module(tp)
        squad_total = sum(p["exp_total"] for p in tp)
        results.append({"team": team, "modulo": bm["module"],
                        "XI_fantapunti": round(bm["val"]),
                        "rosa_totale": round(squad_total),
                        "n_gioc": len(tp), "spesa": sum(p["cost"] for p in tp),
                        "_bm": bm})
    tab = pd.DataFrame(results).sort_values("XI_fantapunti", ascending=False).reset_index(drop=True)
    tab.index = tab.index + 1

    print("="*76)
    print("CLASSIFICA SQUADRE per fantapunti attesi dell'XI titolare (modulo ottimale)")
    print("="*76)
    show = tab[["team", "modulo", "XI_fantapunti", "rosa_totale", "n_gioc", "spesa"]]
    print(show.to_string())
    if unmatched:
        print(f"\n[!] {len(unmatched)} giocatori non abbinati (esclusi): "
              + ", ".join(f"{n}" for _, n in unmatched[:20]))

    MY = C.MY_TEAM
    myrow = tab[tab["team"] == MY]
    pos = int(myrow.index[0]) if len(myrow) else -1
    print("\n" + "="*76)
    print(f"LA MIA SQUADRA: {MY}  —  posizione stimata: {pos}° di {len(tab)}")
    print("="*76)
    if len(myrow):
        bm = myrow.iloc[0]["_bm"]
        print(f"Modulo migliore: {bm['module']}  |  XI fantapunti attesi: {round(bm['val'])}\n")
        print("FORMAZIONE TIPO:")
        starters_ids = set()
        for s in bm["slots"]:
            lab = "/".join(sorted(s["roles"]))
            if s["p"]:
                p = s["p"]; starters_ids.add(p["Id"])
                flags = []
                if p["games_out"] >= 2: flags.append(f"INFORT.~{p['games_out']}g")
                if p["status"] == "ballottaggio": flags.append("ballottaggio")
                if p["pen"] == 1: flags.append("rigorista")
                fl = ("  [" + ", ".join(flags) + "]") if flags else ""
                print(f"  {lab:9s} -> {p['Nome']:20s} {p['exp_total']:6.0f} fpt (pv {p['exp_pv']:.0f}, fm {p['exp_fm']:.2f}){fl}")
            else:
                print(f"  {lab:9s} -> — VUOTO —")
        bench = [p for p in per_team_players[MY] if p["Id"] not in starters_ids]
        bench.sort(key=lambda p: -p["exp_total"])
        print("\nPANCHINA (per fantapunti attesi):")
        for p in bench:
            flags = []
            if p["games_out"] >= 2: flags.append(f"INFORT.~{p['games_out']}g")
            if p["status"] == "ballottaggio": flags.append("ball.")
            fl = ("  [" + ", ".join(flags) + "]") if flags else ""
            print(f"  {'/'.join(p['roles']):10s} {p['Nome']:20s} {p['exp_total']:6.0f} fpt{fl}")
        # module comparison for my team
        print("\nConfronto moduli (XI fantapunti) per la mia rosa:")
        comp = sorted(((m, best_lineup(per_team_players[MY], m)) for m in C.MANTRA_MODULES),
                      key=lambda x: -x[1]["val"])
        for m, r in comp:
            print(f"  {m:9s} {round(r['val']):5d}  (schierabili {r['filled']}/11)")


if __name__ == "__main__":
    run()
