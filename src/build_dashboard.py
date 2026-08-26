"""
build_dashboard.py — Phase 8: self-contained local auction demo (offline, double-click).

Single HTML file, all projections embedded as JSON, all recompute in vanilla JS. Priority is
functionality/clarity, not aesthetics. Auction is random-order; you can change module every
matchday; Mantra has NO Classic per-role minimums.

Iterazione 3 — tracking multi-squadra:
  - Ogni acquisto è assegnato a una delle 12 squadre della lega (la mia = MY_TEAM).
  - Panoramica realistica di tutte le rose → prezzi "competition-aware".
  - Pannelli: copertura ruoli, miglior modulo + XI (dropdown), Lista della spesa dinamica,
    Piano budget, Occasioni & trappole, Squadre della lega, Top per ruolo, ricerca+verdetto.

Run: python src/build_dashboard.py
"""
from __future__ import annotations
import os
import json
import pandas as pd

try:
    from . import config as C
except ImportError:
    import config as C


def _records() -> list[dict]:
    df = pd.read_csv(os.path.join(C.PROCESSED_DIR, "player_values.csv"))
    df = df.where(pd.notna(df), None)
    # merge Monte Carlo (probable price, bargain prob, substitutability verdict)
    mc_path = os.path.join(C.PROCESSED_DIR, "montecarlo.csv")
    mc = {}
    if os.path.exists(mc_path):
        mdf = pd.read_csv(mc_path)
        for _, r in mdf.iterrows():
            mc[int(r["Id"])] = {
                "simMed": int(r.get("sim_price_median") or 0),
                "bargain": float(r.get("prob_bargain") or 0),
                "verdict": r.get("verdict") or "",
                "safe": int(r.get("safe_max_spend") or 0),
            }
    def nz(x, d=0.0):
        """Safe numeric: NaN/None -> default (the `x or 0` idiom is broken for NaN, which is truthy)."""
        try:
            v = float(x)
            return d if v != v else v      # v != v is True only for NaN
        except (TypeError, ValueError):
            return d

    recs = []
    for _, r in df.iterrows():
        roles = [x for x in str(r.get("mantra_roles") or "").split(";") if x]
        if not roles and r.get("role"):
            roles = [r["role"]]
        pid = int(r["Id"])
        m = mc.get(pid, {})
        recs.append({
            "id": pid, "name": r["Nome"], "team": r["team"],
            "role": r["role"], "roles": roles, "macro": r["macro"],
            "pts": round(nz(r["exp_total"]), 1),
            "pv": round(nz(r["exp_pv"]), 1),
            "fm": round(nz(r["exp_fm"]), 2),
            "g": round(nz(r.get("exp_goals")), 1),
            "a": round(nz(r.get("exp_assists")), 1),
            "var": round(nz(r["VAR"]), 1),
            "price": int(nz(r["exp_price"], 1)),
            "vpp": round(nz(r.get("value_per_price")), 2),
            "risk": round(nz(r["risk"]), 3),
            "fvm": int(nz(r["FVM"])),
            "buy": round(nz(r["buyability"]), 1),
            "new": int(nz(r.get("is_newcomer"))),
            "conf": r.get("prior_confidence") if isinstance(r.get("prior_confidence"), str) else "",
            "simMed": int(nz(m.get("simMed"))), "bargain": round(nz(m.get("bargain")), 2),
            "mcVerdict": m.get("verdict") if isinstance(m.get("verdict"), str) else "",
            "inj": int(nz(r.get("games_out"))),
            "injNote": r.get("injury_note") if isinstance(r.get("injury_note"), str) else "",
            "status": r.get("starter_status") if isinstance(r.get("starter_status"), str) else "",
            "pen": int(nz(r.get("pen_rank"))), "sp": int(nz(r.get("setpiece"))),
        })
    return recs


def _role_mandatory() -> dict:
    """Max, over all modules, of slots that REQUIRE exactly this role (mono-role slots).
    ≈ how many *starters* of that role you may need to field simultaneously."""
    mand = {r: 0 for r in C.MANTRA_ROLES}
    for slots in C.MANTRA_MODULES.values():
        cnt = {r: 0 for r in C.MANTRA_ROLES}
        for s in slots:
            if len(s) == 1:
                cnt[next(iter(s))] += 1
        for r in C.MANTRA_ROLES:
            mand[r] = max(mand[r], cnt[r])
    return mand


def _role_ideal() -> dict:
    mand = _role_mandatory()
    return {r: (C.GK_TARGET if r == "Por" else max(2, mand[r] + 1)) for r in C.MANTRA_ROLES}


def _role_starters() -> dict:
    """Starter-quality slots to aim for per role (GK=1; others = mandatory max, ≥1)."""
    mand = _role_mandatory()
    return {r: (1 if r == "Por" else max(1, mand[r])) for r in C.MANTRA_ROLES}


def build() -> str:
    data = _records()
    default_module = "4-3-3"
    comp_path = os.path.join(C.OUTPUTS_DIR, "module_comparison.csv")
    if os.path.exists(comp_path):
        default_module = str(pd.read_csv(comp_path).iloc[0]["module"])
    cfg = {
        "BUDGET": C.BUDGET, "ROSTER_MIN": C.ROSTER_MIN, "ROSTER_MAX": C.ROSTER_MAX,
        "GK_TARGET": C.GK_TARGET, "N_PARTICIPANTS": C.N_PARTICIPANTS,
        "TEAMS": C.LEAGUE_TEAMS, "MY_TEAM": C.MY_TEAM,
        "MODULES": list(C.MANTRA_MODULES.keys()), "DEFAULT_MODULE": default_module,
        "MODULE_SLOTS": {m: [sorted(list(s)) for s in slots]
                         for m, slots in C.MANTRA_MODULES.items()},
        "MANTRA_ROLES": C.MANTRA_ROLES, "ROLE_IDEAL": _role_ideal(),
        "ROLE_STARTERS": _role_starters(),
        "ROLE_NAMES": {
            "Por": "Portiere", "Dc": "Dif. centrale", "B": "Braccetto", "Dd": "Terzino dx",
            "Ds": "Terzino sx", "E": "Esterno basso", "M": "Mediano", "C": "Centrale",
            "W": "Ala", "T": "Trequartista", "A": "Att. esterno", "Pc": "Punta centrale"},
    }
    html = (TEMPLATE
            .replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
            .replace("/*__CONFIG__*/", json.dumps(cfg, ensure_ascii=False)))
    out = os.path.join(C.DASHBOARD_DIR, "auction_dashboard.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[dashboard] {len(data)} giocatori, {len(C.LEAGUE_TEAMS)} squadre -> {out}")
    print(f"[dashboard] aprilo col doppio click (offline). Modulo default: {default_module}")
    return out


TEMPLATE = r"""<!doctype html>
<html lang="it"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fantacalcio Mantra 2026/27 — Assistente d'Asta</title>
<style>
  *{box-sizing:border-box}
  body{font-family:system-ui,Arial,sans-serif;margin:0;background:#eef0f2;color:#1a1a1a;font-size:13px}
  header{background:#0b3d2e;color:#fff;padding:8px 14px;position:sticky;top:0;z-index:20}
  header h1{margin:0;font-size:15px}
  .wrap{padding:10px 14px;max-width:1300px;margin:0 auto}
  .panel{background:#fff;border:1px solid #d7d7d7;border-radius:6px;padding:10px 12px;margin-bottom:10px}
  .panel h2{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.03em;color:#0b3d2e}
  .cards{display:flex;flex-wrap:wrap;gap:8px}
  .card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:6px 12px;min-width:105px}
  .card .big{font-size:20px;font-weight:700}.card .lbl{font-size:10px;color:#666;text-transform:uppercase}
  input,select,button{font-size:13px;padding:5px 7px;border:1px solid #bbb;border-radius:4px}
  button{cursor:pointer;background:#0b3d2e;color:#fff;border:none}
  button.sec{background:#4a5}button.gray{background:#888}
  button.mini{padding:2px 7px;font-size:11px}
  table{border-collapse:collapse;width:100%;font-size:12px}
  th,td{padding:3px 6px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap}
  th:first-child,td:first-child,th.l,td.l{text-align:left}
  th{background:#f2f2f2;position:sticky;top:0}
  tr:hover td{background:#f7fbff}
  .tag{display:inline-block;padding:1px 6px;border-radius:10px;font-size:10px;color:#fff}
  .risk-lo{background:#2e8b57}.risk-mid{background:#c98a00}.risk-hi{background:#c0392b}
  .new{color:#b5651d;font-weight:600}
  .rolegrid{display:flex;flex-wrap:wrap;gap:6px}
  .rc{border:1px solid #ccc;border-radius:6px;padding:4px 8px;min-width:92px;background:#fafafa}
  .rc b{font-size:15px}.rc .nm{font-size:10px;color:#555}
  .rc.ok{background:#e7f5ec;border-color:#8ec9a6}.rc.miss{background:#fdeaea;border-color:#e2a0a0}
  .rc.part{background:#fff7e6;border-color:#e0c072}
  .rolebtns button{background:#e8eaed;color:#222;margin:2px;border:1px solid #ccc}
  .rolebtns button.active{background:#0b3d2e;color:#fff}
  .searchcard{border:2px solid #0b3d2e;border-radius:8px;padding:10px;margin-top:8px}
  .score{font-size:28px;font-weight:800}.lbl-badge{font-size:12px;padding:2px 9px;border-radius:12px;color:#fff}
  .muted{color:#777;font-size:11px}
  .flex{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .maxbid{font-weight:700;color:#0b3d2e}
  .empty{color:#c0392b;font-style:italic}
  .banner{display:flex;gap:14px;flex-wrap:wrap;align-items:baseline}.banner .m{font-size:20px;font-weight:800;color:#0b3d2e}
  .cols{display:flex;gap:10px;flex-wrap:wrap}.col{flex:1;min-width:330px}
  .mine{background:#eaf6ee!important;font-weight:600}
  .warn{color:#b02a2a}.good{color:#0b7a3b}
  a{color:#0b3d2e;cursor:pointer}
</style></head><body>
<header><h1>⚽ Fantacalcio Mantra 2026/27 — Assistente d'Asta (offline)</h1></header>
<div class="wrap">

  <div class="cards" id="cards"></div>

  <div class="panel">
    <h2>Registra acquisto</h2>
    <div class="flex">
      <input id="searchInput" list="playerlist" placeholder="Cerca giocatore…" style="min-width:220px" autocomplete="off">
      <datalist id="playerlist"></datalist>
      <input id="priceInput" type="number" min="1" placeholder="prezzo" style="width:80px">
      <label>Squadra:</label><select id="teamSel"></select>
      <button onclick="buy()">Registra acquisto</button>
      <button class="gray" onclick="resetAll()">Reset</button>
    </div>
    <div class="muted">Assegna ogni giocatore alla squadra che l'ha preso (default: la tua). Serve a stimare i prezzi in base alla concorrenza reale.</div>
    <div id="searchCard"></div>
  </div>

  <div class="panel">
    <h2>Copertura ruoli Mantra (la mia rosa)</h2>
    <div class="rolegrid" id="roleCov"></div>
    <div class="muted" style="margin-top:6px">Obiettivo: una <b>coppia</b> per ruolo (3 per Dc/E/M/Por). Un giocatore multi-ruolo conta in ciascuno. Verde=ok, giallo=1, rosso=scoperto.</div>
  </div>

  <div class="cols">
    <div class="col panel">
      <h2>Miglior modulo per la MIA rosa</h2>
      <div class="banner" id="bestMineBanner"></div>
      <div class="flex" style="margin-top:6px"><label>Vedi modulo:</label><select id="moduleSel" onchange="render()"></select></div>
      <div id="xiTable" style="margin-top:6px"></div>
    </div>
    <div class="col panel">
      <h2>Lista della spesa — chi prendere ora</h2>
      <div id="shopping"></div>
      <div class="muted">Priorità = riempire i tuoi slot da titolare mancanti, poi le riserve. <b>Prob.</b>=prezzo probabile (concorrenza). <b>Max TE</b>=tetto per te.</div>
    </div>
  </div>

  <div class="cols">
    <div class="col panel">
      <h2>Piano budget / strategia crediti</h2>
      <div id="budgetPlan"></div>
    </div>
    <div class="col panel">
      <h2>Occasioni &amp; trappole</h2>
      <div id="dealsTraps"></div>
    </div>
  </div>

  <div class="panel">
    <h2>Migliori da prendere per ruolo — prezzo e consiglio personalizzato</h2>
    <div class="rolebtns" id="roleBtns"></div>
    <div id="roleTable" style="margin-top:8px;overflow-x:auto"></div>
    <div class="muted"><b>Prob.</b>=prezzo probabile (concorrenza avversari). <b>Max TE</b>=tetto per te (budget+ruoli). 🆕=newcomer.</div>
  </div>

  <div class="panel">
    <h2>Squadre della lega (crediti &amp; rose)</h2>
    <div id="teamsTable" style="overflow-x:auto"></div>
    <div class="muted">Clicca una squadra per vederne la rosa. Alimenta la stima dei prezzi (chi ha budget e cerca un ruolo fa salire il prezzo).</div>
  </div>

  <div class="muted">Riferimento esatto (ILP + simulazione) in Python: <code>optimize.py</code>, <code>auction.py</code>, <code>montecarlo.py</code>. La demo usa un'euristica veloce.</div>
</div>

<script>
const DATA = /*__DATA__*/;
const CFG = /*__CONFIG__*/;
const byId={}; DATA.forEach(p=>byId[p.id]=p);
// "starter benchmark" per ruolo = Fpt dell'ultimo giocatore di livello-titolare della lega in
// quel ruolo (rank = partecipanti x titolari-del-ruolo). Serve a capire se un giocatore che
// possiedo è davvero da titolare o solo un rincalzo. Costante (dipende solo dai dati).
const STARTER_BENCH={};
CFG.MANTRA_ROLES.forEach(r=>{
  const arr=DATA.filter(p=>p.roles.includes(r)).map(p=>p.pts).sort((a,b)=>b-a);
  const thr=(CFG.ROLE_STARTERS&&CFG.ROLE_STARTERS[r])||1;
  const rank=Math.min(arr.length-1, Math.round(CFG.N_PARTICIPANTS*thr));
  STARTER_BENCH[r]= arr.length? arr[rank]*0.90 : 0;   // 0.90 = un po' di tolleranza
});
let state = load();
let CTX=null, selectedRole='Pc', expandedTeams={}, dealsOpen=null;

function load(){
  let s; try{ s=JSON.parse(localStorage.getItem('fanta_auction_v3')); }catch(e){}
  if(s && s.teams) return s;
  // fresh init (also soft-migrate old {mine,others} → my team / discard others' identities)
  const teams={}; CFG.TEAMS.forEach(t=>teams[t]=[]);
  try{ const old=JSON.parse(localStorage.getItem('fanta_auction'));
    if(old&&old.mine) old.mine.forEach(x=>teams[CFG.MY_TEAM].push(x)); }catch(e){}
  return {mineName:CFG.MY_TEAM, teams};
}
function save(){ localStorage.setItem('fanta_auction_v3', JSON.stringify(state)); }

// ---------- derived ----------
function teamList(t){ return state.teams[t]||[]; }
function teamSpent(t){ return teamList(t).reduce((s,x)=>s+x.price,0); }
function teamBudget(t){ return CFG.BUDGET - teamSpent(t); }
function teamCoverage(t){ const c={}; CFG.MANTRA_ROLES.forEach(r=>c[r]=0);
  teamList(t).forEach(x=>{ const p=byId[x.id]; if(p) p.roles.forEach(r=>{if(r in c)c[r]++;}); }); return c; }
function teamNeeds(t,roles){ const c=teamCoverage(t); return roles.some(r=>(c[r]||0)<(CFG.ROLE_IDEAL[r]||2)); }
function spent(){ return teamSpent(state.mineName); }
function budget(){ return teamBudget(state.mineName); }
function myList(){ return teamList(state.mineName); }
function myPlayers(){ return myList().map(x=>byId[x.id]).filter(Boolean); }
function allTaken(){ const s=new Set(); CFG.TEAMS.forEach(t=>teamList(t).forEach(x=>s.add(x.id))); return s; }
function unavailable(){ return allTaken(); }
function available(){ const u=allTaken(); return DATA.filter(p=>!u.has(p.id)); }
function roleCoverage(){ return teamCoverage(state.mineName); }

function buildCtx(){
  const u=allTaken(); const avail=DATA.filter(p=>!u.has(p.id));
  const bud=budget(); const slotsLeft=Math.max(1, CFG.ROSTER_MIN - myList().length);
  const fairRemaining=Math.max(1, CFG.BUDGET*slotsLeft/CFG.ROSTER_MIN);
  const pressure=Math.max(0.4, Math.min(2.5, bud/fairRemaining));
  const availRole={}; CFG.MANTRA_ROLES.forEach(r=>availRole[r]=0);
  avail.forEach(p=>p.roles.forEach(r=>{ if(r in availRole && p.var>0) availRole[r]++; }));
  CTX={avail,bud,slotsLeft,pressure,availRole,cov:roleCoverage()};
  return CTX;
}

// ---------- competition-aware probable price ----------
function competition(p){ let n=0;
  CFG.TEAMS.forEach(t=>{ if(t===state.mineName) return;
    if(teamList(t).length>=CFG.ROSTER_MAX) return;
    if(teamBudget(t) >= Math.max(3, p.price*0.5) && teamNeeds(t, p.roles)) n++; });
  return n;
}
function probPrice(p){ const comp=competition(p);
  const demand=Math.max(0.8, Math.min(1.7, 0.8 + 0.06*comp));
  return {price:Math.max(1, Math.round(p.price*demand)), comp}; }

// ---------- my max bid & personalization ----------
function maxBidGeneral(p){ const c=CTX; const spendable=Math.max(1, c.bud-(c.slotsLeft-1));
  let sc=1; p.roles.forEach(r=>{ const need=Math.max(0,(CFG.ROLE_IDEAL[r]||2)-(c.cov[r]||0));
    const av=c.availRole[r]||0; const f= need>0 ? (1+0.35*Math.max(0,1-av/Math.max(1,need*3))) : 1; sc=Math.max(sc,f); });
  return Math.max(1, Math.min(Math.round(p.price*c.pressure*1.15*sc), spendable)); }
// role acquisition tier given what I ALREADY own:
//   'starter' = I still need a starter in this role   → paga da titolare
//   'backup'  = starters presi, mi serve una RISERVA  → solo economico (idealmente il vice)
//   'full'    = ruolo completo                         → solo a sconto
function starterThr(role){ return (CFG.ROLE_STARTERS&&CFG.ROLE_STARTERS[role])||1; }
// best Fpt still buyable in a role (players not taken by anyone)
function bestAvailPts(role){ const c=CTX||buildCtx(); let mx=0;
  c.avail.forEach(p=>{ if(p.roles.includes(role)&&p.pts>mx) mx=p.pts; }); return mx; }
// ADATTIVO: soglia-titolare = min(soglia assoluta del ruolo, miglior giocatore ancora sul mercato).
// Se i top del ruolo sono già stati comprati, la soglia scende e il miglior disponibile rimasto
// conta come titolare (così non ti spinge a inseguire un livello che non esiste più).
function effBench(role){ const b=STARTER_BENCH[role]||0, av=bestAvailPts(role);
  return av>0 ? Math.min(b, av) : 0; }
function myStartersInRole(role){ const b=effBench(role);
  return myPlayers().filter(p=>p.roles.includes(role) && p.pts>=b).length; }
function tierForRole(role){
  const owned=CTX.cov[role]||0;
  // still need a real STARTER if I don't yet own enough starter-quality players in this role,
  // even if I already own weaker bodies there (they don't "cover" the starter need)
  if(myStartersInRole(role) < starterThr(role)) return 'starter';
  if(owned < (CFG.ROLE_IDEAL[role]||2)) return 'backup';
  return 'full'; }
function myStarterTeamFor(role){ const m=myPlayers().filter(p=>p.roles.includes(role)).sort((a,b)=>b.pts-a.pts);
  return m.length? m[0].team : null; }
function bestTier(p){ const rank={starter:2,backup:1,full:0}; let tier='full', tRole=null;
  p.roles.forEach(r=>{ const t=tierForRole(r); if(!tRole||rank[t]>rank[tier]){tier=t;tRole=r;} });
  return {tier,tRole}; }

function personal(p){ const c=CTX; const gen=maxBidGeneral(p);
  const {tier,tRole}=bestTier(p);
  const spendable=Math.max(1, c.bud-(c.slotsLeft-1));
  let pmax;
  if(tier==='starter') pmax=Math.min(gen,spendable);
  else if(tier==='backup'){ const cap=Math.max(1, Math.round(0.5*c.bud/c.slotsLeft)); pmax=Math.min(gen,spendable,cap); }
  else pmax=Math.min(spendable, Math.round(gen*0.4));
  if(myList().length>=CFG.ROSTER_MAX) pmax=0;
  const needRoles=p.roles.filter(r=>(c.cov[r]||0)<(CFG.ROLE_IDEAL[r]||2));
  return {gen, pmax:Math.max(0,pmax), tier, tRole, needRoles}; }

function verdict(p){ const c=CTX; const pe=personal(p); const pp=probPrice(p);
  if(unavailable().has(p.id)) return {t:'GIÀ PRESO', col:'#777', pe, pp};
  if(myList().length>=CFG.ROSTER_MAX) return {t:'Rosa piena', col:'#777', pe, pp};
  if(c.bud<=0) return {t:'Crediti finiti', col:'#b02a2a', pe, pp};
  if(pe.tier==='full') return {t:'Ruolo completo — solo a sconto (≤'+pe.pmax+')', col:'#8a6d00', pe, pp};
  if(pe.tier==='backup'){ let msg='RISERVA per '+pe.tRole+' — prendi economico (max '+pe.pmax+')';
    if(pe.tRole==='Por' && p.team===myStarterTeamFor('Por')) msg='VICE del tuo portiere ('+p.team+') — ideale come riserva (max '+pe.pmax+')';
    return {t:msg, col:'#8a6d00', pe, pp}; }
  if(pe.pmax<1) return {t:'Budget insufficiente', col:'#b02a2a', pe, pp};
  // tier === 'starter'
  const affWord = pp.price<=pe.pmax ? '' : ' — probabile che vada oltre il tuo tetto';
  if(pp.comp===0) return {t:'LIBERO: nessuno te lo contende, offri il minimo (max '+pe.pmax+')', col:'#0b7a3b', pe, pp};
  if(p.buy>=68) return {t:'DA PRENDERE (titolare '+pe.tRole+', max '+pe.pmax+')'+affWord, col: pp.price<=pe.pmax?'#0b7a3b':'#8a6d00', pe, pp};
  if(p.buy>=48) return {t:'Buon titolare ('+pe.tRole+', max '+pe.pmax+')'+affWord, col:'#3b7a0b', pe, pp};
  return {t:'Alternativa da titolare (max '+pe.pmax+')'+affWord, col:'#8a6d00', pe, pp}; }
function buyability(p){ const c=CTX; const pool=c.avail.filter(q=>q.roles.some(r=>p.roles.includes(r)));
  const vmax=Math.max(1,...pool.map(q=>Math.max(0,q.var||0))); const varN=Math.max(0,(p.var||0))/vmax;
  const need=bestTier(p).tier!=='full'?1:0.3; const lowRisk=1-Math.min(1,p.risk);
  const mb=maxBidGeneral(p); const afford=c.bud>=mb?1:0.5;
  return Math.round(100*(0.40*varN+0.24*need+0.18*lowRisk+0.18*afford)); }
function riskTag(r){const c=r<0.18?'risk-lo':r<0.33?'risk-mid':'risk-hi';const t=r<0.18?'basso':r<0.33?'medio':'alto';return `<span class="tag ${c}">${t}</span>`;}
function injTag(p){ return p.inj>=2 ? ` <span class="tag risk-hi" title="${(p.injNote||'').replace(/"/g,'')}">🚑 ~${p.inj}g</span>` : ''; }
function statusTag(p){ let h='';
  if(p.status==='titolare') h+=' <span class="tag" style="background:#2e8b57" title="Titolare probabile">TIT</span>';
  else if(p.status==='ballottaggio') h+=' <span class="tag" style="background:#c98a00" title="In ballottaggio">BALL</span>';
  if(p.pen===1) h+=' <span class="tag" style="background:#0b3d2e" title="Rigorista">⚽</span>';
  else if(p.pen>1) h+=` <span class="muted" title="Rigorista ${p.pen}ª scelta">⚽${p.pen}</span>`;
  if(p.sp) h+=' <span class="muted" title="Calci da fermo">◎</span>';
  return h; }

// ---------- lineup ----------
function bestLineup(module, players){ const slots=CFG.MODULE_SLOTS[module].map((roles,i)=>({i,roles,player:null}));
  const pool=players.slice().sort((a,b)=>b.pts-a.pts); const used=new Set();
  const order=slots.map((s,idx)=>idx).sort((x,y)=>{
    const ex=pool.filter(p=>p.roles.some(r=>slots[x].roles.includes(r))).length;
    const ey=pool.filter(p=>p.roles.some(r=>slots[y].roles.includes(r))).length;
    return slots[x].roles.length-slots[y].roles.length || ex-ey; });
  for(const idx of order){ const s=slots[idx];
    for(const p of pool){ if(used.has(p.id))continue; if(p.roles.some(r=>s.roles.includes(r))){ s.player=p; used.add(p.id); break; } } }
  const val=slots.reduce((a,s)=>a+(s.player?s.player.pts:0),0);
  return {slots:slots.sort((a,b)=>a.i-b.i), val:Math.round(val), filled:slots.filter(s=>s.player).length}; }
function bestModuleFor(players){ let best=null; for(const m of CFG.MODULES){ const r=bestLineup(m,players);
  const sc=r.filled*1e6+r.val; if(!best||sc>best.sc) best={module:m,...r,sc}; } return best; }

// ---------- actions ----------
function findPlayer(txt){ txt=txt.trim().toLowerCase(); if(!txt)return null;
  let m=DATA.find(p=>(p.name+' '+p.team).toLowerCase()===txt); if(m)return m;
  const cs=DATA.filter(p=>p.name.toLowerCase().includes(txt)); return cs.length?cs.sort((a,b)=>b.pts-a.pts)[0]:null; }
function buy(){ const p=findPlayer(document.getElementById('searchInput').value);
  if(!p){alert('Giocatore non trovato');return;} if(unavailable().has(p.id)){alert('Già registrato');return;}
  const pr=parseInt(document.getElementById('priceInput').value); if(!pr||pr<1){alert('Inserisci il prezzo');return;}
  const t=document.getElementById('teamSel').value; state.teams[t].push({id:p.id,price:pr});
  document.getElementById('searchInput').value=''; document.getElementById('priceInput').value='';
  save(); render(); }
function removeBuy(team,id){ state.teams[team]=state.teams[team].filter(x=>x.id!==id); save(); render(); }
function resetAll(){ if(confirm('Azzerare tutta l\'asta?')){ const teams={}; CFG.TEAMS.forEach(t=>teams[t]=[]); state={mineName:CFG.MY_TEAM,teams}; save(); render(); } }
function quickPick(id){ const p=byId[id]; document.getElementById('searchInput').value=p.name+' '+p.team; document.getElementById('priceInput').focus(); renderSearchCard(); }
function toggleTeam(t){ expandedTeams[t]=!expandedTeams[t]; renderTeams(); }

// ---------- render ----------
function render(){ buildCtx(); renderCards(); renderRoleCov(); renderBestMine(); renderShopping();
  renderBudgetPlan(); renderDeals(); renderRoleBtns(); renderRoleTable(); renderTeams();
  renderSearchCard(); renderDatalist(); initSelects(); }

function initSelects(){ const ts=document.getElementById('teamSel');
  if(!ts.dataset.init){ ts.innerHTML=CFG.TEAMS.map(t=>`<option value="${t}" ${t===CFG.MY_TEAM?'selected':''}>${t}${t===CFG.MY_TEAM?' (io)':''}</option>`).join(''); ts.dataset.init='1'; }
  const ms=document.getElementById('moduleSel');
  if(!ms.dataset.init){ ms.innerHTML=CFG.MODULES.map(m=>`<option value="${m}">${m}</option>`).join(''); ms.value=CFG.DEFAULT_MODULE; ms.dataset.init='1'; } }

function renderCards(){ const B=budget(),n=myList().length,sl=CTX.slotsLeft;
  document.getElementById('cards').innerHTML=`
    <div class="card"><div class="big">${B}</div><div class="lbl">Crediti residui (${state.mineName})</div></div>
    <div class="card"><div class="big">${spent()}</div><div class="lbl">Spesi</div></div>
    <div class="card"><div class="big">${n} / ${CFG.ROSTER_MIN}-${CFG.ROSTER_MAX}</div><div class="lbl">In rosa</div></div>
    <div class="card"><div class="big">${Math.max(0,CFG.ROSTER_MIN-n)}</div><div class="lbl">Slot minimi rimasti</div></div>
    <div class="card"><div class="big">${sl>0?(B/sl).toFixed(0):'—'}</div><div class="lbl">Credito medio / slot</div></div>`; }

function renderRoleCov(){ const cov=CTX.cov;
  document.getElementById('roleCov').innerHTML=CFG.MANTRA_ROLES.map(r=>{ const n=cov[r]||0,id=CFG.ROLE_IDEAL[r]||2;
    const cls=n===0?'miss':(n<2?'part':'ok');
    return `<div class="rc ${cls}"><b>${n}</b> <span class="muted">/${id}</span><div class="nm">${r} · ${CFG.ROLE_NAMES[r]||r}</div></div>`; }).join(''); }

function renderBestMine(){ const best=bestModuleFor(myPlayers());
  document.getElementById('bestMineBanner').innerHTML=`<span>Miglior modulo ora:</span> <span class="m">${best.module}</span> <span>schierabili <b>${best.filled}/11</b></span> <span>valore XI <b>${best.val}</b></span>`;
  const sel=document.getElementById('moduleSel'); const mod=sel&&sel.value?sel.value:best.module;
  document.getElementById('xiTable').innerHTML=renderXI(mod, myPlayers()); }
function renderXI(module, players){ const bl=bestLineup(module,players);
  let h=`<table><thead><tr><th class="l">Ruolo</th><th class="l">Giocatore</th><th>Fpt</th></tr></thead><tbody>`;
  bl.slots.forEach(s=>{ const lab=s.roles.join('/');
    if(s.player) h+=`<tr><td class="l">${lab}</td><td class="l">${s.player.name} <span class="muted">${s.player.team}</span>${s.player.new?' 🆕':''}</td><td>${s.player.pts}</td></tr>`;
    else h+=`<tr><td class="l">${lab}</td><td class="l empty">— serve un ${lab} —</td><td>-</td></tr>`; });
  return h+'</tbody></table>'; }

function renderShopping(){ const best=bestModuleFor(myPlayers()); const avail=CTX.avail; const used=new Set(); const items=[];
  best.slots.forEach(s=>{
    if(!s.player){ const cand=avail.filter(p=>!used.has(p.id)&&p.roles.some(r=>s.roles.includes(r))).sort((a,b)=>b.var-a.var)[0];
      if(cand){ used.add(cand.id); items.push({p:cand, slot:s.roles.join('/'), tag:'titolare mancante', pr:0}); } }
    else { // slot filled but by a weak (below-benchmark) player -> suggest an upgrade
      const bench=Math.min(...s.roles.map(r=>STARTER_BENCH[r]||0));
      if(s.player.pts < bench){
        const cand=avail.filter(p=>!used.has(p.id)&&p.roles.some(r=>s.roles.includes(r))&&p.pts>s.player.pts).sort((a,b)=>b.var-a.var)[0];
        if(cand){ used.add(cand.id); items.push({p:cand, slot:s.roles.join('/'), tag:'upgrade titolare', pr:0.5}); }
      } } });
  // reserves: roles still under ideal after my players
  CFG.MANTRA_ROLES.forEach(r=>{ let need=(CFG.ROLE_IDEAL[r]||2)-(CTX.cov[r]||0);
    // subtract starters already suggested that cover r
    items.forEach(it=>{ if(it.p.roles.includes(r)) need--; });
    const myTeam = r==='Por'? myStarterTeamFor('Por') : null;
    let k=0; while(need>0 && k<2){ const cand=avail.filter(p=>!used.has(p.id)&&p.roles.includes(r))
        .sort((a,b)=> (r==='Por'?(Number(b.team===myTeam)-Number(a.team===myTeam)):0) || b.vpp-a.vpp || a.price-b.price)[0];
      if(!cand)break; used.add(cand.id); items.push({p:cand, slot:r, tag:'riserva', pr:1}); need--; k++; } });
  items.sort((a,b)=> a.pr-b.pr || b.p.var-a.p.var);
  let h=`<table><thead><tr><th class="l">Giocatore</th><th class="l">Per</th><th class="l">Tipo</th><th>Fpt</th><th>Prob.</th><th>Max TE</th><th></th></tr></thead><tbody>`;
  items.slice(0,14).forEach(it=>{ const v=verdict(it.p);
    const tagcol=it.tag==='titolare mancante'?'#0b3d2e':(it.tag==='upgrade titolare'?'#b58900':'#888');
    h+=`<tr><td class="l">${it.p.name}${it.p.new?' 🆕':''}${statusTag(it.p)}${injTag(it.p)}</td><td class="l">${it.slot}</td>
      <td class="l"><span class="tag" style="background:${tagcol}">${it.tag}</span></td>
      <td>${it.p.pts}</td><td>${v.pp.price}</td><td class="maxbid">${v.pe.pmax}</td>
      <td><button class="mini" onclick="quickPick(${it.p.id})">→</button></td></tr>`; });
  h+='</tbody></table>'; if(!items.length)h='<div class="muted good">Rosa completa per il miglior modulo! 🎉</div>';
  document.getElementById('shopping').innerHTML=h; }

function renderBudgetPlan(){ const B=budget(), n=myList().length, need=Math.max(0,CFG.ROSTER_MIN-n);
  const el=document.getElementById('budgetPlan');
  if(need<=0){ el.innerHTML=`<div class="good">Hai raggiunto ${n} giocatori. Crediti residui: <b>${B}</b> (puoi arrivare a ${CFG.ROSTER_MAX}).</div>`; return; }
  const keepMin=need-1; const spendable=Math.max(0,B-keepMin);
  const perSlot=(B/need);
  // tier plan: how many "top" (>=25) you can still afford leaving ~5 avg for the rest
  const avgRest=6; let tops=Math.floor((B-(need*avgRest))/ (40-avgRest)); tops=Math.max(0,Math.min(tops,need));
  let msg='';
  if(perSlot<4) msg=`<span class="warn">Budget teso: ~${perSlot.toFixed(0)} cr/slot. Punta su tanti giocatori a basso prezzo e 1-2 scommesse.</span>`;
  else if(perSlot>18) msg=`<span class="warn">Hai molti crediti per pochi slot (~${perSlot.toFixed(0)} cr/slot): puoi/ DEVI investire su top, non avanzare crediti!</span>`;
  else msg=`<span class="good">Equilibrio ok: ~${perSlot.toFixed(0)} cr/slot.</span>`;
  el.innerHTML=`
    <table><tbody>
      <tr><td class="l">Crediti residui</td><td><b>${B}</b></td></tr>
      <tr><td class="l">Slot ancora da riempire (min)</td><td><b>${need}</b></td></tr>
      <tr><td class="l">Da tenere per gli altri slot (≥1 cad.)</td><td>${keepMin}</td></tr>
      <tr><td class="l">Spendibili ORA su un giocatore</td><td class="maxbid">${spendable}</td></tr>
      <tr><td class="l">Budget medio / slot</td><td>${perSlot.toFixed(0)}</td></tr>
      <tr><td class="l">Top (~30-60cr) ancora sostenibili</td><td>~${tops}</td></tr>
    </tbody></table><div style="margin-top:6px">${msg}</div>`; }

function renderDeals(){ const avail=CTX.avail;
  // AFFARI per ruolo: top 5 per rapporto valore/prezzo (V/P), tra i disponibili, esclusi
  // infortuni lunghi. Un dropdown per ruolo (aperti di default quelli che mi servono).
  if(dealsOpen===null){ dealsOpen=new Set(); }   // tutti chiusi di default (compatto) — clicca per aprire
  let h='<div class="muted" style="margin-bottom:6px">Migliori rapporti <b>valore/prezzo</b> per ruolo (top 5). Clicca un ruolo per espanderlo.</div>';
  CFG.MANTRA_ROLES.forEach(role=>{
    const list=avail.filter(p=>p.roles.includes(role)&&p.vpp>0&&p.price>=2&&p.inj<4)
      .sort((a,b)=>b.vpp-a.vpp).slice(0,5);
    if(!list.length) return;
    const need=(CTX.cov[role]||0)<(CFG.ROLE_IDEAL[role]||2);
    const open=dealsOpen.has(role)?' open':'';
    h+=`<details${open} ontoggle="if(this.open)dealsOpen.add('${role}');else dealsOpen.delete('${role}')" style="margin-bottom:3px">
      <summary style="cursor:pointer"><b>${role}</b> · ${CFG.ROLE_NAMES[role]||role} <span class="muted">(${list.length})${need?' — ti serve':' — coperto'}</span></summary>
      <table><thead><tr><th class="l">Giocatore</th><th class="l">Sq</th><th>Fpt</th><th>VAR</th><th>Prezzo</th><th>V/P</th><th>Prob.</th><th></th></tr></thead><tbody>`;
    list.forEach(p=>{ h+=`<tr><td class="l">${p.name}${statusTag(p)}${injTag(p)}</td><td class="l">${p.team}</td>
      <td>${p.pts}</td><td>${p.var}</td><td>${p.price}</td><td class="maxbid">${p.vpp}</td><td>${probPrice(p).price}</td>
      <td><button class="mini" onclick="quickPick(${p.id})">→</button></td></tr>`; });
    h+='</tbody></table></details>';
  });
  // TRAPPOLE: costoso ma sostituibile / prezzo probabile molto sopra il valore (1 per ruolo)
  const seenT=new Set(); const traps=[];
  avail.filter(p=>p.price>=15 && (p.mcVerdict==='Sostituibile' || probPrice(p).price> p.price*1.3))
    .sort((a,b)=>b.price-a.price)
    .forEach(p=>{ if(!seenT.has(p.role)){ seenT.add(p.role); traps.push(p); } });
  h+='<div style="margin-top:8px"><b class="warn">Da non sovrapagare</b>';
  h+='<table><thead><tr><th class="l">Giocatore</th><th class="l">R</th><th>Prezzo att.</th><th>Prob.</th><th class="l">Nota</th></tr></thead><tbody>';
  traps.forEach(p=>{ h+=`<tr><td class="l"><a onclick="quickPick(${p.id})">${p.name}</a></td><td class="l">${p.role}</td><td>${p.price}</td><td>${probPrice(p).price}</td><td class="l">${p.mcVerdict||'coda prezzo alta'}</td></tr>`; });
  h+='</tbody></table></div>';
  document.getElementById('dealsTraps').innerHTML=h; }

function renderRoleBtns(){ document.getElementById('roleBtns').innerHTML=CFG.MANTRA_ROLES.map(r=>{
  const ok=(CTX.cov[r]||0)>=(CFG.ROLE_IDEAL[r]||2)?' ✓':'';
  return `<button class="${r===selectedRole?'active':''}" onclick="selectedRole='${r}';renderRoleBtns();renderRoleTable()">${r}${ok}</button>`; }).join(''); }
function renderRoleTable(){ const u=allTaken();
  const tier=tierForRole(selectedRole);
  let rows=DATA.filter(p=>!u.has(p.id)&&p.roles.includes(selectedRole));
  if(tier==='starter'){ rows.sort((a,b)=>b.var-a.var); }
  else { const myTeam=myStarterTeamFor(selectedRole);   // backup/full: economici, e per i Por il vice
    rows.sort((a,b)=> (selectedRole==='Por'?(Number(b.team===myTeam)-Number(a.team===myTeam)):0) || b.vpp-a.vpp || a.price-b.price); }
  rows=rows.slice(0,10);
  const have=CTX.cov[selectedRole]||0;
  const stOwned=myStartersInRole(selectedRole);
  let note;
  if(tier==='starter'){
    const topGone = bestAvailPts(selectedRole) < (STARTER_BENCH[selectedRole]||0);
    if(topGone)
      note = `I <b>titolari top</b> di questo ruolo sono già stati acquistati → prendi comunque il <b>miglior disponibile rimasto</b> (è il tuo titolare).`;
    else if(have>0 && stOwned<starterThr(selectedRole))
      note = `Hai <b>${have}</b> giocatore/i qui ma <b>nessuno da titolare</b> di livello → punta a un <b>TITOLARE più forte</b> (prezzo pieno).`;
    else
      note = `Ti serve un <b>titolare</b> in questo ruolo → ecco i migliori per valore.`;
  }
  else if(tier==='backup'){ const mt=myStarterTeamFor(selectedRole);
    note=`Hai già il/i titolare/i → cerca una <b>RISERVA economica</b>${selectedRole==='Por'&&mt?`, idealmente il <b>vice del ${mt}</b>`:''}. Ordinati per convenienza (valore/prezzo), non per valore assoluto.`; }
  else note=`Ruolo <b>completo</b>: prendi altri solo a sconto.`;
  let h=`<div class="muted" style="margin-bottom:4px">Ruolo <b>${CFG.ROLE_NAMES[selectedRole]||selectedRole}</b> — in rosa: ${have}/${CFG.ROLE_IDEAL[selectedRole]||2} · ${note}</div>`;
  h+=`<table><thead><tr><th class="l">#</th><th class="l">Giocatore</th><th class="l">Sq</th><th>Fpt</th><th>VAR</th><th>Risk</th><th>Prob.</th><th>Max TE</th><th>Contesa</th><th class="l">Consiglio</th><th></th></tr></thead><tbody>`;
  rows.forEach((p,i)=>{ const v=verdict(p);
    h+=`<tr><td class="l">${i+1}</td><td class="l">${p.name}${p.new?' <span class="new">🆕</span>':''}${statusTag(p)}${injTag(p)}</td><td class="l">${p.team}</td>
      <td>${p.pts}</td><td>${p.var}</td><td>${riskTag(p.risk)}</td><td>${v.pp.price}</td><td class="maxbid">${v.pe.pmax}</td>
      <td>${v.pp.comp}</td><td class="l" style="color:${v.col};white-space:normal;max-width:220px">${v.t}</td>
      <td><button class="mini" onclick="quickPick(${p.id})">→</button></td></tr>`; });
  h+='</tbody></table>'; if(!rows.length)h='<div class="muted">Nessun giocatore disponibile per questo ruolo.</div>';
  document.getElementById('roleTable').innerHTML=h; }

function renderTeams(){ let h=`<table><thead><tr><th class="l">Squadra</th><th>Crediti</th><th>Giocatori</th><th>Spesi</th><th class="l">Ruoli scoperti</th></tr></thead><tbody>`;
  const teams=CFG.TEAMS.slice().sort((a,b)=> (a===state.mineName?-1:b===state.mineName?1:0));
  teams.forEach(t=>{ const cov=teamCoverage(t); const miss=CFG.MANTRA_ROLES.filter(r=>(cov[r]||0)===0);
    h+=`<tr class="${t===state.mineName?'mine':''}"><td class="l"><a onclick="toggleTeam('${t.replace(/'/g,"\\'")}')">${t}${t===state.mineName?' (io)':''}</a></td>
      <td>${teamBudget(t)}</td><td>${teamList(t).length}</td><td>${teamSpent(t)}</td>
      <td class="l muted">${miss.length? miss.join(', '):'—'}</td></tr>`;
    if(expandedTeams[t]){ const pl=teamList(t);
      h+=`<tr><td colspan="5" class="l">${ pl.length? pl.map(x=>{const p=byId[x.id]; return `${p?p.name:x.id} (${p?p.roles.join(';'):''}, ${x.price}cr) <a onclick="removeBuy('${t.replace(/'/g,"\\'")}',${x.id})">✕</a>`;}).join(' · ') : '<span class="muted">nessun acquisto</span>' }</td></tr>`; } });
  h+='</tbody></table>'; document.getElementById('teamsTable').innerHTML=h; }

function renderDatalist(){ const u=allTaken();
  document.getElementById('playerlist').innerHTML=DATA.filter(p=>!u.has(p.id)).map(p=>`<option value="${p.name} ${p.team}">`).join(''); }
function renderSearchCard(){ const p=findPlayer(document.getElementById('searchInput').value);
  const el=document.getElementById('searchCard'); if(!p){el.innerHTML='';return;}
  const s=buyability(p), v=verdict(p);
  const alts=DATA.filter(q=>!unavailable().has(q.id)&&q.id!==p.id&&q.roles.some(r=>p.roles.includes(r))).sort((a,b)=>b.var-a.var).slice(0,3);
  el.innerHTML=`<div class="searchcard">
    <div class="flex" style="justify-content:space-between">
      <div><b style="font-size:15px">${p.name}</b> <span class="muted">${p.team} · ${p.roles.join(';')} ${p.new?'<span class="new">🆕</span>':''}</span>${statusTag(p)}${injTag(p)}</div>
      <div style="text-align:center"><div class="score" style="color:${v.col}">${s}</div><div class="muted">acquistabilità</div></div></div>
    ${p.inj>=2?`<div class="warn" style="margin-top:4px">🚑 Infortunato — salta ~${p.inj} giornate. ${p.injNote||''} (proiezioni e prezzo già ridotti)</div>`:''}
    <div class="flex" style="margin-top:4px">
      <span class="lbl-badge" style="background:${v.col}">${v.t}</span>
      <span>Max per TE: <span class="maxbid" style="font-size:16px">${v.pe.pmax}</span></span>
      <span class="muted">prezzo probabile ${v.pp.price} · contesa da ${v.pp.comp} squadre · atteso ${p.price}</span></div>
    <table style="margin-top:6px"><tbody>
      <tr><td class="l">Fantapunti attesi</td><td>${p.pts}</td><td class="l">Presenze attese</td><td>${p.pv}</td></tr>
      <tr><td class="l">Gol / Assist attesi</td><td>${p.g} / ${p.a}</td><td class="l">VAR</td><td>${p.var}</td></tr>
      <tr><td class="l">Rischio</td><td>${riskTag(p.risk)}</td><td class="l">FVM</td><td>${p.fvm}</td></tr>
    </tbody></table>
    <div class="muted" style="margin-top:4px">Alternative stesso ruolo: ${alts.map(a=>`${a.name} (max ${personal(a).pmax})`).join(' · ')||'—'}</div></div>`; }

document.getElementById('searchInput').addEventListener('input', renderSearchCard);
render();
</script>
</body></html>
"""

if __name__ == "__main__":
    build()
