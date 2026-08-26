"""
report.py — Phase 10: final report + Excel export.

Assembles outputs/report.md (sections A–G) and outputs/player_rankings.xlsx (multi-sheet)
from the processed artifacts. Also runs a small sensitivity analysis via the ILP.

Run: python src/report.py   (assumes value.py, optimize.py, montecarlo.py already ran)
"""
from __future__ import annotations
import os
import pandas as pd

try:
    from . import config as C
    from . import optimize as OPT
except ImportError:
    import config as C
    import optimize as OPT

ROLE_ORDER = ["Por", "Dc", "B", "Dd", "Ds", "E", "M", "C", "W", "T", "A", "Pc"]
ROLE_NAMES = {"Por": "Portieri", "Dc": "Difensori centrali", "B": "Braccetti",
              "Dd": "Terzini destri", "Ds": "Terzini sinistri", "E": "Esterni bassi",
              "M": "Mediani", "C": "Centrali", "W": "Ali", "T": "Trequartisti",
              "A": "Attaccanti/Ali", "Pc": "Punte centrali"}


def _md(df, cols):
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_markdown(index=False)


def run() -> str:
    pv = pd.read_csv(os.path.join(C.PROCESSED_DIR, "player_values.csv"))
    mc = pd.read_csv(os.path.join(C.PROCESSED_DIR, "montecarlo.csv")) \
        if os.path.exists(os.path.join(C.PROCESSED_DIR, "montecarlo.csv")) else None
    comp = pd.read_csv(os.path.join(C.OUTPUTS_DIR, "module_comparison.csv"))
    roster = pd.read_csv(os.path.join(C.OUTPUTS_DIR, "optimal_roster.csv"))
    pv["role_ord"] = pv["role"].apply(lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else 99)

    L = []
    W = L.append
    W("# Fantacalcio Mantra 2026/27 — Report finale\n")
    W("_Generato da `src/report.py`. Budget 500, 12 partecipanti, rosa 25–30. "
      "Metodo: proiezioni ML (temporal validation) → VAR/prezzi → ottimizzazione ILP → "
      "simulazione Monte Carlo. Dettagli e limiti in `outputs/data_audit.md` e "
      "`outputs/model_metrics.md`._\n")

    # ---- A. Player ranking --------------------------------------------------
    W("## A. Ranking giocatori (top 40)\n")
    a_cols = ["Nome", "team", "role", "exp_price", "exp_total", "exp_pv", "exp_goals",
              "exp_assists", "VAR", "value_per_price", "risk", "max_bid_static", "buyability"]
    W(_md(pv.sort_values("exp_total", ascending=False).head(40), a_cols) + "\n")
    W("_Colonne: exp_total=fantapunti stagionali attesi, VAR=valore sopra il rimpiazzo, "
      "max_bid_static=offerta max pre-asta, buyability=indice 0-100. Ranking completo nel file "
      "Excel `player_rankings.xlsx`._\n")

    # ---- B. Ranking per role ------------------------------------------------
    W("## B. Ranking per ruolo Mantra (top 8)\n")
    for role in ROLE_ORDER:
        sub = pv[pv["role"] == role].sort_values("exp_total", ascending=False).head(8)
        if not len(sub):
            continue
        W(f"### {role} — {ROLE_NAMES.get(role, role)}\n")
        W(_md(sub, ["Nome", "team", "exp_price", "exp_total", "VAR",
                    "max_bid_static", "risk", "buyability"]) + "\n")

    # ---- C. Value picks per role -------------------------------------------
    W("## C. Migliori value pick per ruolo (VAR/credito)\n")
    W("_I portieri sono strutturalmente efficienti (1 titolare): confrontali solo tra loro._\n")
    for role in ROLE_ORDER:
        sub = pv[(pv["role"] == role) & (pv["VAR"] > 0) & (pv["exp_price"] >= 3)] \
            .sort_values("value_per_price", ascending=False).head(4)
        if not len(sub):
            continue
        W(f"**{role}**: " + ", ".join(
            f"{r.Nome} ({r.team}, {int(r.exp_price)}cr, VAR {r.VAR:.0f}, v/p {r.value_per_price:.2f})"
            for r in sub.itertuples()) + "\n")

    # ---- D. Optimal module --------------------------------------------------
    W("\n## D. Modulo ottimale (selezione statistica)\n")
    W(_md(comp, ["module", "starter_points", "bench_points", "cost", "roster_size",
                 "depth_ratio", "avg_risk", "overall_score"]) + "\n")
    best = comp.iloc[0]
    second = comp.iloc[1]
    W(f"**Modulo migliore: `{best['module']}`** (overall score {best['overall_score']:.0f}, "
      f"XI {best['starter_points']:.0f}, costo {int(best['cost'])}, "
      f"profondità {best['depth_ratio']:.2f}, rischio {best['avg_risk']:.2f}).\n")
    W(f"Perché: massimizza il valore complessivo della rosa entro 500 crediti. Il distacco dagli "
      f"altri moduli è contenuto (2° `{second['module']}` a {second['overall_score']:.0f}): con una "
      f"rosa flessibile 25–30 e i prezzi correnti, più moduli sono quasi equivalenti. "
      f"**Questa è un'informazione strategica utile**: conviene costruire una rosa che consenta di "
      f"passare tra i moduli migliori (es. {best['module']} ↔ {second['module']}) in base a come "
      f"va l'asta, invece di vincolarsi a uno solo. Il modulo vincente sfrutta l'abbondanza di "
      f"esterni/centrocampisti ad alto VAR e la relativa economicità dei difensori.\n")

    # ---- E. Optimal roster --------------------------------------------------
    W("## E. Rosa ottimale (500 crediti)\n")
    st = roster[roster["is_starter"] == 1]
    bn = roster[roster["is_starter"] == 0]
    W(f"Modulo `{roster['module'].iloc[0]}` · {len(roster)} giocatori · "
      f"spesa {int(roster['paid'].sum())} crediti.\n")
    W("**Titolari:**\n")
    W(_md(st, ["Nome", "team", "role", "value", "paid", "VAR", "risk"]) + "\n")
    W("**Panchina:**\n")
    W(_md(bn, ["Nome", "team", "role", "value", "paid"]) + "\n")

    # ---- F. Alternatives ----------------------------------------------------
    W("## F. Alternative per i titolari chiave (stesso ruolo, disponibili)\n")
    owned = set(roster["Nome"])
    for r in st.sort_values("value", ascending=False).head(8).itertuples():
        role = r.role
        alts = pv[(pv["role"] == role) & (~pv["Nome"].isin(owned))] \
            .sort_values("exp_total", ascending=False).head(3)
        alt_s = ", ".join(f"{a.Nome} ({a.team}, {int(a.exp_price)}cr)" for a in alts.itertuples())
        W(f"- **{r.Nome}** ({role}) → {alt_s or '—'}")
    W("")

    # ---- G. Sensitivity -----------------------------------------------------
    W("\n## G. Sensitivity analysis\n")
    pool = OPT.load_pool()
    base = OPT.solve_module(best["module"], pool)
    base_score = base["overall_score"]
    W(f"Baseline (modulo {best['module']}): valore rosa {base['total_points']:.0f}, "
      f"XI {base['starter_points']:.0f}, costo {base['cost']}.\n")
    scenarios = []

    # G1: top player taken by an opponent
    top_player_id = int(pv.sort_values("exp_total", ascending=False).iloc[0]["Id"])
    top_name = pv.sort_values("exp_total", ascending=False).iloc[0]["Nome"]
    s1 = OPT.solve_module(best["module"], pool, taken_ids={top_player_id})
    scenarios.append((f"{top_name} preso da un avversario", s1))

    # G2: budget -50 and +50
    s2 = OPT.solve_module(best["module"], pool, budget=C.BUDGET - 50)
    scenarios.append(("Budget 450 (−50)", s2))
    s3 = OPT.solve_module(best["module"], pool, budget=C.BUDGET + 50)
    scenarios.append(("Budget 550 (+50)", s3))

    # G3: price inflation +20% on all
    pool_inf = pool.copy()
    pool_inf["cost"] = (pool_inf["cost"] * 1.2).round().clip(lower=1).astype(int)
    s4 = OPT.solve_module(best["module"], pool_inf)
    scenarios.append(("Inflazione prezzi +20%", s4))

    rows = []
    for name, s in scenarios:
        if s.get("feasible"):
            rows.append({"scenario": name, "valore_rosa": round(s["total_points"], 0),
                         "XI": round(s["starter_points"], 0), "costo": s["cost"],
                         "Δ_valore_vs_base": round(s["total_points"] - base["total_points"], 0)})
        else:
            rows.append({"scenario": name, "valore_rosa": "INFEASIBLE"})
    W(_md(pd.DataFrame(rows), ["scenario", "valore_rosa", "XI", "costo", "Δ_valore_vs_base"]) + "\n")
    W("_Interpretazione: la perdita di valore se un top player va a un avversario indica quanto è "
      "'indispensabile' (vedi anche `outputs/auction_strategy.md`). La sensibilità al budget "
      "quantifica il valore marginale di ogni credito._\n")

    # ---- write ---
    with open(os.path.join(C.OUTPUTS_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    # Excel export (multi-sheet)
    xlsx = os.path.join(C.OUTPUTS_DIR, "player_rankings.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        full = pv.sort_values("exp_total", ascending=False)
        full.drop(columns=["role_ord"]).to_excel(xl, sheet_name="Ranking", index=False)
        for role in ROLE_ORDER:
            sub = pv[pv["role"] == role].sort_values("exp_total", ascending=False)
            if len(sub):
                sub.drop(columns=["role_ord"]).to_excel(xl, sheet_name=role, index=False)
        roster.to_excel(xl, sheet_name="Rosa_ottimale", index=False)
        comp.to_excel(xl, sheet_name="Moduli", index=False)
        if mc is not None:
            mc.to_excel(xl, sheet_name="MonteCarlo", index=False)

    print(f"[report] -> outputs/report.md e outputs/player_rankings.xlsx")
    return os.path.join(C.OUTPUTS_DIR, "report.md")


if __name__ == "__main__":
    run()
