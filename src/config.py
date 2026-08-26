"""
config.py — Central configuration for the Fantacalcio Mantra 2026/27 Auction Optimizer.

EVERY tunable parameter lives here so the whole pipeline is reproducible and adjustable
without touching logic. Values that are ASSUMPTIONS (not derived from the provided data)
are flagged with `# ASSUMPTION`.

League: FANTAISACA/ISACA (Mantra). Source of rules: REGOLE FANTAISACA.pdf.
"""
from __future__ import annotations
import os

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
# Project root = parent of this file's directory (src/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = ROOT                      # raw xlsx are kept in place in the project root
DATA_DIR = os.path.join(ROOT, "data")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
DASHBOARD_DIR = os.path.join(ROOT, "dashboard")

for _d in (DATA_DIR, EXTERNAL_DIR, PROCESSED_DIR, OUTPUTS_DIR, DASHBOARD_DIR):
    os.makedirs(_d, exist_ok=True)

# Raw data files (read in place). AUTO-DISCOVERED: drop any
# `Statistiche_Fantacalcio_Stagione_YYYY_YY_Italia.xlsx` into the project root and it is
# picked up automatically — no code change needed. More historical seasons = more temporal
# transition pairs = better validation (even players no longer active add training signal).
import glob as _glob
import re as _re


def _discover_stats_files() -> dict:
    out = {}
    pat = _re.compile(r"Statistiche_Fantacalcio_Stagione_(\d{4})_(\d{2})_Italia\.xlsx$", _re.I)
    for path in _glob.glob(os.path.join(RAW_DIR, "Statistiche_Fantacalcio_Stagione_*_Italia.xlsx")):
        m = pat.search(os.path.basename(path))
        if m:
            season = f"{m.group(1)}-{m.group(2)}"      # e.g. 2023-24
            out[season] = os.path.basename(path)
    # Fallback to the known three if discovery finds nothing (defensive).
    if not out:
        out = {
            "2023-24": "Statistiche_Fantacalcio_Stagione_2023_24_Italia.xlsx",
            "2024-25": "Statistiche_Fantacalcio_Stagione_2024_25_Italia.xlsx",
            "2025-26": "Statistiche_Fantacalcio_Stagione_2025_26_Italia.xlsx",
        }
    return dict(sorted(out.items()))


STATS_FILES = _discover_stats_files()


def _discover_quotes() -> str:
    """Newest Quotazioni_*.xlsx by modification time (so a freshly downloaded listone wins)."""
    files = _glob.glob(os.path.join(RAW_DIR, "Quotazioni_Fantacalcio_Stagione_*.xlsx"))
    if not files:
        return "Quotazioni_Fantacalcio_Stagione_2026_27.xlsx"
    files.sort(key=os.path.getmtime)
    return os.path.basename(files[-1])


QUOTES_FILE = _discover_quotes()
TARGET_SEASON = "2026-27"          # season we predict / build the roster for
HISTORICAL_SEASONS = list(STATS_FILES.keys())              # sorted ascending
SEASON_ORDER = HISTORICAL_SEASONS + [TARGET_SEASON]

# In every workbook the real header is on the 2nd row (row 0 is a title banner).
EXCEL_HEADER_ROW = 1

# --------------------------------------------------------------------------------------
# League rules (from REGOLE FANTAISACA.pdf) -> optimizer parameters
# --------------------------------------------------------------------------------------
BUDGET = 500                       # crediti totali per l'asta
N_PARTICIPANTS = 12                # 12 società (10 storiche + Daddi & Nitto). Configurable.

# Squadre della lega (fantallenatori) — usate dalla demo per tracciare gli acquisti per squadra
# reale e stimare prezzi in base alla concorrenza. MY_TEAM = la squadra dell'utente.
LEAGUE_TEAMS = [
    "Topeldaccia FC", "Real Malaspina", "S.U.Sanna Rosalia", "FC Palagonia Mondello",
    "Dallas", "AC Borgo Febio", "F.C. BORGO NUOVO", "ZENit Sampolo",
    "Monte Piddirinu Tzitzato", "Centro Edilizia Popolare", "AC ALBERGHERIA", "Atletico Mondello",
]
MY_TEAM = "Topeldaccia FC"
ROSTER_MIN = 25                    # "Da 25 a 30 giocatori in rosa"
ROSTER_MAX = 30
BENCH_MIN_GK = 1                   # "Panchina: 12 giocatori di cui almeno un portiere"

# --- Composizione rosa: modello MANTRA (NON Classic) ---------------------------------
# In Mantra NON esistono minimi fissi per ruolo (niente "8 dif, 8 cen, 6 att"). Serve solo
# poter schierare SEMPRE un modulo (con un minimo di rotazione). Il modulo è un RIFERIMENTO
# per costruire la rosa, non un vincolo rigido: lo si può cambiare ogni giornata.
#
# Vincoli reali usati dall'ottimizzatore:
#   - totale rosa ∈ [ROSTER_MIN, ROSTER_MAX]
#   - portieri = GK_TARGET (unico floor posizionale vero: devi schierarne uno + panchina)
#   - per il modulo di riferimento: abbastanza giocatori per coprire l'XI + un buffer di
#     rotazione (DEPTH_BUFFER), calcolato dai ruoli effettivamente "forzati" dal modulo.
#   - NESSUN massimo per macro-ruolo (libertà totale nella composizione).
GK_TARGET = 3                       # portieri in rosa (1 titolare + panchina, ≥1 in panchina)
# Buffer di rotazione oltre ai titolari "forzati" dal modulo, per macro-ruolo di movimento.
# A=2 garantisce almeno un minimo di punte di riserva per poter passare ai moduli a 2 punte.
DEPTH_BUFFER = {"D": 2, "C": 2, "A": 2}


def module_forced_macro(module: str) -> dict:
    """Quanti giocatori di ciascun macro-ruolo l'XI del modulo FORZA (slot mono-macro).
    Gli slot 'flessibili' (es. {W,A} = C o A) non forzano alcun macro."""
    forced = {"P": 0, "D": 0, "C": 0, "A": 0}
    for slot in MANTRA_MODULES[module]:
        macros = {ROLE_TO_MACRO[r] for r in slot}
        if len(macros) == 1:
            forced[next(iter(macros))] += 1
    return forced


def module_macro_min(module: str) -> dict:
    """Minimi per macro-ruolo per il modulo dato: portieri = GK_TARGET; movimento = forzati +
    buffer di rotazione. Sono FLOOR di profondità, non quote in stile Classic."""
    forced = module_forced_macro(module)
    return {
        "P": GK_TARGET,
        "D": forced["D"] + DEPTH_BUFFER.get("D", 1),
        "C": forced["C"] + DEPTH_BUFFER.get("C", 1),
        "A": forced["A"] + DEPTH_BUFFER.get("A", 1),
    }

# --------------------------------------------------------------------------------------
# Mantra roles taxonomy (atomic roles actually present in the data)
# --------------------------------------------------------------------------------------
# Confirmed from the 2026/27 quotazioni + stats files:
#   Por | Dc, B, Dd, Ds, E | M, C, W, T | A, Pc
# B = "braccetto" (wide centre-back in a back-3). E = esterno/wing-back.
MANTRA_ROLES = ["Por", "Dc", "B", "Dd", "Ds", "E", "M", "C", "W", "T", "A", "Pc"]

# Which macro role (R column) each atomic Mantra role belongs to.
ROLE_TO_MACRO = {
    "Por": "P",
    "Dc": "D", "B": "D", "Dd": "D", "Ds": "D", "E": "D",
    "M": "C", "C": "C", "W": "C", "T": "C",
    "A": "A", "Pc": "A",
}
# NB: 'E' is coded as macro D (it is in the Difensori sheet) but is eligible for wing
# midfield slots in several modules (see MANTRA_MODULES). 'T'/'W' are macro C but eligible
# for attack slots. Eligibility is governed by the module slot sets below, not the macro.

# --------------------------------------------------------------------------------------
# Mantra modules -> starting-XI slot definitions
# --------------------------------------------------------------------------------------
# Each module is a list of 11 slots; each slot is the SET of atomic Mantra roles that may
# fill it. SOURCE: the official "Moduli Mantra — Edizione 2026/2027" image provided by the
# user (Moduli Mantra.jpg) — transcribed slot-by-slot. NOTE: this edition has NO 5-defender
# modules. Slot labels map as: DC={Dc}, DC/B={Dc,B}, DD={Dd}, DS={Ds}, E={E}, E/W={E,W},
# M={M}, C={C}, M/C={M,C}, C/T={C,T}, W={W}, T={T}, W/T={W,T}, W/A={W,A}, T/A={T,A},
# A/PC={A,Pc}, T/A/PC={T,A,Pc}.
def _s(*roles):  # slot helper
    return frozenset(roles)

MANTRA_MODULES = {
    "3-4-3":   [_s("Por"), _s("Dc"),_s("Dc"),_s("Dc","B"),
                _s("E"),_s("M","C"),_s("C"),_s("E"),
                _s("W","A"),_s("A","Pc"),_s("W","A")],
    "3-4-1-2": [_s("Por"), _s("Dc"),_s("Dc"),_s("Dc","B"),
                _s("E"),_s("M","C"),_s("C"),_s("E"),
                _s("T"),_s("A","Pc"),_s("A","Pc")],
    "3-4-2-1": [_s("Por"), _s("Dc"),_s("Dc"),_s("Dc","B"),
                _s("M"),_s("M","C"),_s("E"),_s("E","W"),
                _s("T"),_s("T","A"),_s("A","Pc")],
    "3-5-2":   [_s("Por"), _s("Dc"),_s("Dc"),_s("Dc","B"),
                _s("M"),_s("M","C"),_s("E"),_s("E","W"),_s("C"),
                _s("A","Pc"),_s("A","Pc")],
    "3-5-1-1": [_s("Por"), _s("Dc"),_s("Dc"),_s("Dc","B"),
                _s("M"),_s("M"),_s("C"),_s("E","W"),_s("E","W"),
                _s("T","A"),_s("A","Pc")],
    "4-3-3":   [_s("Por"), _s("Dd"),_s("Dc"),_s("Dc"),_s("Ds"),
                _s("M","C"),_s("M"),_s("C"),
                _s("W","A"),_s("A","Pc"),_s("W","A")],
    "4-3-1-2": [_s("Por"), _s("Dd"),_s("Dc"),_s("Dc"),_s("Ds"),
                _s("M","C"),_s("M"),_s("C"),
                _s("T"),_s("T","A","Pc"),_s("A","Pc")],
    "4-4-2":   [_s("Por"), _s("Dd"),_s("Dc"),_s("Dc"),_s("Ds"),
                _s("M","C"),_s("C"),_s("E"),_s("E","W"),
                _s("A","Pc"),_s("A","Pc")],
    "4-1-4-1": [_s("Por"), _s("Dd"),_s("Dc"),_s("Dc"),_s("Ds"),
                _s("M"),_s("C","T"),_s("T"),_s("E","W"),_s("W"),
                _s("A","Pc")],
    "4-4-1-1": [_s("Por"), _s("Dd"),_s("Dc"),_s("Dc"),_s("Ds"),
                _s("M"),_s("C"),_s("E","W"),_s("E","W"),
                _s("T","A"),_s("A","Pc")],
    "4-2-3-1": [_s("Por"), _s("Dd"),_s("Dc"),_s("Dc"),_s("Ds"),
                _s("M"),_s("M","C"),
                _s("W","T"),_s("T"),_s("W","A"),_s("A","Pc")],
}

# --------------------------------------------------------------------------------------
# Mantra scoring rules (fantavoto). ASSUMPTION: standard Fantagazzetta Mantra scoring;
# the PDF does not define custom bonus/malus. Used for component recomposition and
# clean-sheet modelling; the primary value target also uses the historical Fm directly.
# --------------------------------------------------------------------------------------
SCORING = {
    "goal_P": 3.0, "goal_D": 3.0, "goal_C": 3.0, "goal_A": 3.0,  # gol (bonus uguale nei nostri dati Fm)
    "assist": 1.0,
    "penalty_scored": 3.0,      # R+
    "penalty_missed": -3.0,     # R-
    "penalty_saved": 3.0,       # Rp (GK)
    "goal_conceded_P": -1.0,    # Gs per portiere, ogni gol subito
    "yellow": -0.5,             # Amm
    "red": -1.0,                # Esp
    "own_goal": -2.0,           # Au
    "clean_sheet_P": 1.0,       # imbattibilità portiere
    # Modificatore difesa e clean-sheet per DC non modellati a livello partita (dato mancante).
}

# --------------------------------------------------------------------------------------
# League-strength coefficients for foreign newcomers (prior league -> Serie A).
# Multiplier applied to per-90 attacking output when projecting to Serie A.
# SOURCE/ASSUMPTION: order-of-magnitude values consistent with published football
# analytics league-strength rankings (e.g. relative goal/xG conversion between leagues).
# These are PRIORS for players with no Serie A history; flagged as high-uncertainty.
# Adjust in one place here. >1 means the source league is stronger than Serie A.
# --------------------------------------------------------------------------------------
LEAGUE_STRENGTH = {
    "Premier League": 1.20, "La Liga": 1.05, "Bundesliga": 1.02, "Serie A": 1.00,
    "Ligue 1": 0.95, "Primeira Liga": 0.80, "Eredivisie": 0.78,
    "Championship": 0.75, "Serie B": 0.62, "Liga Argentina": 0.70,
    "Brasileirao": 0.75, "MLS": 0.65, "Super Lig": 0.72, "Jupiler Pro League": 0.70,
    "Other": 0.65,
}
# Multiplicative discount on EXPECTED MINUTES/availability for a first Serie A season
# (adaptation risk). ASSUMPTION.
NEWCOMER_MINUTES_DISCOUNT = 0.85
# Cap on how many newcomers to web-enrich (prioritised by FVM) to bound web calls.
NEWCOMER_ENRICH_TOP_N = 90

# --------------------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------------------
MAX_MATCHDAYS = 38                 # Serie A season length; Pv is capped at this
QUANTILES = [0.10, 0.50, 0.90]     # pessimistic / median / optimistic
RANDOM_STATE = 42

# Labeled target seasons = every historical season that has at least one season before it
# (so lag features exist). Derived automatically from the discovered seasons.
LABELED_TARGETS = HISTORICAL_SEASONS[1:]          # e.g. [2024-25, 2025-26, ...]
# OOS evaluation: test on the most recent labeled season; train on all earlier labeled ones.
OOS_TEST_TARGET = LABELED_TARGETS[-1] if LABELED_TARGETS else None
OOS_TRAIN_TARGETS = LABELED_TARGETS[:-1] if len(LABELED_TARGETS) > 1 else LABELED_TARGETS
# Final fit uses ALL labeled targets to predict TARGET_SEASON.

# --------------------------------------------------------------------------------------
# Risk scoring weights (0..1 composite; higher = riskier). Configurable.
# --------------------------------------------------------------------------------------
RISK_WEIGHTS = {
    "newcomer": 0.30,          # no Serie A history / first season
    "low_minutes": 0.25,       # low historical Pv (rotation risk)
    "young_age": 0.10,         # age < 21
    "fm_volatility": 0.20,     # inter-season Fm variance
    "prediction_spread": 0.15, # width of quantile band
}
YOUNG_AGE_THRESHOLD = 21

# --------------------------------------------------------------------------------------
# Value / pricing
# --------------------------------------------------------------------------------------
# Replacement level = the (N_PARTICIPANTS * starters_per_role)-th best player per role,
# i.e. the last player likely to be rostered league-wide. Multiplier tunes strictness.
REPLACEMENT_DEPTH_MULT = 1.0
# Expected auction price model: total league spend ≈ N_PARTICIPANTS * BUDGET.
# Price is calibrated from FVM + VAR + scarcity; FVM is an INPUT, not the price.
LEAGUE_TOTAL_BUDGET = N_PARTICIPANTS * BUDGET
# Share of total budget empirically spent per macro role (typical Mantra auctions).
# ASSUMPTION; used to calibrate the price model. Configurable.
BUDGET_SHARE_BY_MACRO = {"P": 0.07, "D": 0.22, "C": 0.33, "A": 0.38}

# --------------------------------------------------------------------------------------
# Optimizer objective weights
# --------------------------------------------------------------------------------------
BENCH_VALUE_WEIGHT = 0.35          # bench contribution weight vs starters in objective
RISK_PENALTY_WEIGHT = 0.10         # penalise expected points by risk*this in objective

# --------------------------------------------------------------------------------------
# Monte Carlo auction simulation
# --------------------------------------------------------------------------------------
MC_N_SIMS = 2000
MC_OPPONENT_PRICE_NOISE = 0.20     # lognormal sigma on opponents' willingness-to-pay
MC_OPPONENT_AGGRESSION = 1.00      # global multiplier on opponents' bids
