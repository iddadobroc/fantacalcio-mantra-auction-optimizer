"""
audit.py — Data audit (Phase 0).

Produces outputs/data_audit.md documenting structure, coverage, quality, duplicates,
missing values, cross-season ID consistency, role vocabulary, team changes, known
incongruences, newcomer count, and — critically — the EXPLICIT list of fields that are
missing from the provided data (with proposed sources). Nothing is invented.

Run:  python src/audit.py
"""
from __future__ import annotations
import os
import pandas as pd

try:
    from . import config as C
    from . import dataio
except ImportError:
    import config as C
    import dataio

COLUMN_LEGEND = {
    "Id": "Identificativo univoco giocatore (Fantacalcio.it), stabile tra stagioni",
    "R": "Ruolo classico (P/D/C/A)",
    "Rm/RM": "Ruolo/i Mantra (es. 'Dd;Ds;E'); B=braccetto, E=esterno",
    "Nome": "Nome giocatore", "Squadra": "Squadra Serie A",
    "Pv": "Partite con Voto (PROXY dei minuti: non sono minuti reali)",
    "Mv": "Media Voto", "Fm": "FantaMedia (fantavoto medio a partita)",
    "Gf": "Gol Fatti", "Gs": "Gol Subiti (portieri/difesa)",
    "Rp": "Rigori Parati", "Rc": "Rigori Calciati/Concessi (legenda ufficiale)",
    "Rplus (R+)": "Rigori Segnati", "Rminus (R-)": "Rigori Sbagliati",
    "Ass": "Assist", "Amm": "Ammonizioni", "Esp": "Espulsioni", "Au": "Autogol",
    "Qt.A / Qt.I": "Quotazione Attuale / Iniziale (classic)",
    "Qt.A M / Qt.I M": "Quotazione Attuale / Iniziale (Mantra)",
    "FVM / FVM M": "Fantacalcio Market Value (classic / Mantra)",
}

MISSING_FIELDS = [
    ("xG (Expected Goals)", "Assente in tutti i file", "FBref, Opta, Understat, SofaScore"),
    ("xA (Expected Assists)", "Assente", "FBref, Opta, Understat"),
    ("Tiri / Key passes / azioni difensive", "Assenti", "FBref, WhoScored, SofaScore"),
    ("Minuti reali giocati", "Solo Pv (partite con voto) come proxy", "Lega Serie A, FBref, Transfermarkt"),
    ("Età / data di nascita", "Assente", "Transfermarkt, Wikipedia, FBref"),
    ("Storico infortuni", "Assente", "Transfermarkt (injury history)"),
    ("Dati campionati esteri (newcomer)", "Assenti", "FBref, Transfermarkt (web-enrichment)"),
    ("Clean sheet per partita", "Derivabile solo in modo aggregato/approssimato per i GK", "Non disponibile a livello partita"),
    ("Prezzi d'asta reali della lega", "Non disponibili; si stima un prezzo atteso", "Storico aste della lega (se esistente)"),
]


def _md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def run() -> str:
    lines: list[str] = []
    W = lines.append
    W("# Data Audit — Fantacalcio Mantra 2026/27\n")
    W(f"_Generato automaticamente da `src/audit.py`. Budget lega: {C.BUDGET} crediti, "
      f"{C.N_PARTICIPANTS} partecipanti, rosa {C.ROSTER_MIN}-{C.ROSTER_MAX}._\n")

    # ---- File inventory -----------------------------------------------------
    W("## 1. Inventario file\n")
    inv = []
    for season, fn in C.STATS_FILES.items():
        p = os.path.join(C.RAW_DIR, fn)
        inv.append({"file": fn, "tipo": "statistiche", "stagione": season,
                    "KB": round(os.path.getsize(p) / 1024, 1)})
    pq = os.path.join(C.RAW_DIR, C.QUOTES_FILE)
    inv.append({"file": C.QUOTES_FILE, "tipo": "quotazioni", "stagione": C.TARGET_SEASON,
                "KB": round(os.path.getsize(pq) / 1024, 1)})
    W(_md_table(pd.DataFrame(inv)) + "\n")

    # ---- Column legend ------------------------------------------------------
    W("## 2. Legenda colonne\n")
    W(_md_table(pd.DataFrame(
        [{"colonna": k, "significato": v} for k, v in COLUMN_LEGEND.items()])) + "\n")

    # ---- Load everything ----------------------------------------------------
    stats = dataio.load_all_stats()
    quotes = dataio.load_quotes()
    ceduti = dataio.load_ceduti()

    # ---- Per-season shape & quality ----------------------------------------
    W("## 3. Struttura e qualità per stagione (statistiche)\n")
    rows = []
    for season in C.STATS_FILES:
        d = stats[stats.season == season]
        rows.append({
            "stagione": season,
            "giocatori": len(d),
            "Id_duplicati": int(d["Id"].duplicated().sum()),
            "Id_mancanti": int(d["Id"].isna().sum()),
            "Pv_mediana": round(d["Pv"].median(), 1),
            "Fm_mediana": round(d["Fm"].median(), 2),
            "%_Mv_zero": round((d["Mv"].fillna(0) == 0).mean() * 100, 1),
        })
    W(_md_table(pd.DataFrame(rows)) + "\n")
    W("_%_Mv_zero_ = quota di giocatori con media voto nulla (di fatto mai schierati "
      "con voto): righe a zero come Osimhen 24/25. Vanno trattate come non-osservazioni.\n")

    # ---- Missing values matrix ---------------------------------------------
    W("## 4. Valori mancanti (statistiche, per colonna e stagione)\n")
    miss = []
    for season in C.STATS_FILES:
        d = stats[stats.season == season]
        row = {"stagione": season}
        for c in dataio.NUMERIC_STATS:
            row[c] = int(d[c].isna().sum())
        miss.append(row)
    W(_md_table(pd.DataFrame(miss)) + "\n")

    # ---- Role vocabulary ----------------------------------------------------
    W("## 5. Vocabolario ruoli\n")
    W("**Macro (R)** conteggi nelle quotazioni 2026/27:\n")
    W(_md_table(quotes["R"].value_counts().rename_axis("R")
                .reset_index(name="n")) + "\n")
    atoms = {}
    for rm in quotes["RM"].dropna():
        for a in dataio.parse_mantra_roles(rm):
            atoms[a] = atoms.get(a, 0) + 1
    atom_df = (pd.DataFrame([{"ruolo_Mantra": k, "n_giocatori": v} for k, v in atoms.items()])
               .sort_values("n_giocatori", ascending=False))
    W("**Ruoli Mantra atomici** (quotazioni 2026/27):\n")
    W(_md_table(atom_df) + "\n")
    W(f"Ruoli Mantra usati dal sistema: `{', '.join(C.MANTRA_ROLES)}`.\n")

    # ---- Teams & changes ----------------------------------------------------
    W("## 6. Squadre per stagione e variazioni\n")
    team_sets = {s: set(stats[stats.season == s]["Squadra"].dropna()) for s in C.STATS_FILES}
    team_sets[C.TARGET_SEASON] = set(quotes["Squadra"].dropna())
    tl = []
    for s in C.SEASON_ORDER:
        ts = team_sets.get(s, set())
        tl.append({"stagione": s, "n_squadre": len(ts),
                   "squadre": ", ".join(sorted(ts))})
    W(_md_table(pd.DataFrame(tl)) + "\n")
    prev = team_sets["2025-26"]
    cur = team_sets[C.TARGET_SEASON]
    W(f"**Nuove squadre 26/27 vs 25/26:** {', '.join(sorted(cur - prev)) or '—'}\n")
    W(f"**Uscite 25/26 -> 26/27:** {', '.join(sorted(prev - cur)) or '—'}\n")
    W("> NB: la lista squadre del file quotazioni 26/27 è la fonte di verità per i "
      "giocatori disponibili all'asta; eventuali discrepanze col vero organico Serie A "
      "vanno risolte dall'utente, non inventate.\n")

    # ---- Cross-season ID consistency ---------------------------------------
    W("## 7. Consistenza Id tra stagioni\n")
    ids = {s: set(stats[stats.season == s]["Id"].dropna().astype(int)) for s in C.STATS_FILES}
    ids[C.TARGET_SEASON] = set(quotes["Id"].dropna().astype(int))
    persist = []
    order = C.SEASON_ORDER
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        inter = ids[a] & ids[b]
        persist.append({"da": a, "a": b, "comuni": len(inter),
                        f"solo_in_{b}": len(ids[b] - ids[a])})
    W(_md_table(pd.DataFrame(persist)) + "\n")

    # ---- Newcomers ----------------------------------------------------------
    newc = ids[C.TARGET_SEASON] - ids["2025-26"]
    q_new = quotes[quotes["Id"].isin(newc)]
    W("## 8. Newcomer (in quotazioni 26/27, assenti dalle stat 25/26)\n")
    W(f"Totale candidati newcomer: **{len(newc)}** su {len(quotes)} giocatori.\n")
    W("Includono: acquisti esteri, promossi dalla Serie B, rientri da prestito, giovani "
      "al debutto. L'origine va determinata via web-enrichment (vedi `newcomers.py`); "
      "non è inferibile dai soli file.\n")
    top_new = (q_new.sort_values("FVM", ascending=False)
               .head(15)[["Nome", "Squadra", "R", "RM", "Qt.A", "FVM"]])
    W("Top 15 newcomer per FVM:\n")
    W(_md_table(top_new) + "\n")

    # ---- Known incongruences -----------------------------------------------
    W("## 9. Incongruenze / anomalie note\n")
    gk_assist = stats[(stats.Rm == "Por") & (stats.Ass.fillna(0) > 0)]
    W(f"- Portieri con assist > 0 (anomalia dati): **{len(gk_assist)}** casi "
      f"(es. { ', '.join(gk_assist['Nome'].dropna().unique()[:5]) }).")
    zero_rows = stats[(stats.Pv.fillna(0) == 0)]
    W(f"- Righe con Pv=0 (giocatori mai a voto, es. trasferiti a stagione in corso): "
      f"**{len(zero_rows)}** — da escludere dalle osservazioni di training.")
    W(f"- Ceduti nel file quotazioni 26/27: **{len(ceduti)}** giocatori (quotazione ~1, "
      "da escludere dal pool d'asta).")
    W("- Il ruolo macro 'A' non compare come categoria separata in alcune viste storiche; "
      "gestito via mappatura ruoli Mantra.\n")

    # ---- Value ranges -------------------------------------------------------
    W("## 10. Range di valore (quotazioni 26/27)\n")
    vr = quotes.groupby("R").agg(
        n=("FVM", "size"), FVM_min=("FVM", "min"), FVM_med=("FVM", "median"),
        FVM_max=("FVM", "max"), QtA_med=("Qt.A", "median"), QtA_max=("Qt.A", "max"),
    ).reset_index()
    W(_md_table(vr) + "\n")

    # ---- Missing fields -----------------------------------------------------
    W("## 11. Campi mancanti (NON inventati) e fonti proposte\n")
    mf = pd.DataFrame([{"campo": a, "stato": b, "fonte proposta": c}
                       for a, b, c in MISSING_FIELDS])
    W(_md_table(mf) + "\n")
    W("### Implicazioni metodologiche\n")
    W("- **Minuti**: si usa `Pv` come proxy della titolarità; dichiarato come limite.\n"
      "- **xG/xA**: gol/assist modellati dai valori storici effettivi (più rumorosi). "
      "xG/xA usati solo per i newcomer via web.\n"
      "- **Solo 3 stagioni**: la validazione temporale ha 2 sole coppie di transizione; "
      "i modelli complessi rischiano overfit, quindi si confrontano OOS con baseline "
      "semplici e si sceglie il migliore.\n")

    md = "\n".join(lines) + "\n"
    out = os.path.join(C.OUTPUTS_DIR, "data_audit.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[audit] scritto {out} ({len(md)} char)")
    print(f"[audit] newcomer candidati: {len(newc)} | ceduti: {len(ceduti)}")
    return out


if __name__ == "__main__":
    run()
