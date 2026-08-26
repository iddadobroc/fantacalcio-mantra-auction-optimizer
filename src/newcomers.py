"""
newcomers.py — Phase 2: foreign / new-to-Serie-A player handling.

A "genuine newcomer" = a 2026/27 auction player with ZERO prior Serie A minutes across the
three available stats seasons (`is_newcomer==1` in players_master.csv). These include foreign
signings, Serie B promotions, loan returns and youth debutants — the origin is NOT inferable
from the provided files, so it must be enriched from the web (user's chosen approach).

This module:
  1. detect() -> the newcomer worklist (top-N by FVM), written to
     data/external/newcomers_to_enrich.csv for the web-enrichment step to fill.
  2. load_enriched() -> reads data/external/newcomers_enriched.csv (schema below) if present,
     validating and attaching a per-player confidence flag. Missing stays missing.
  3. build_priors() -> per-Id Serie A projection PRIORS for newcomers:
       - if enriched web data exists: translate prior-league per-90 output to Serie A using
         LEAGUE_STRENGTH coefficients + a first-season minutes discount;
       - else: a role/FVM-based fallback prior flagged HIGH uncertainty.
     Nothing is invented silently: every prior carries `prior_source` and `prior_confidence`.

Enrichment CSV schema (data/external/newcomers_enriched.csv), one row per Id:
  Id, matched_name, source, prev_club, prev_league, age, minutes_prev, matches_prev,
  goals_prev, assists_prev, xg_prev, xa_prev, rating_prev, transfer_type, transfer_fee_eur_m,
  confidence, notes
Leave a cell blank if the datum cannot be found — do NOT guess.

Run:  python src/newcomers.py            # writes the worklist
      python src/newcomers.py --priors   # builds priors from whatever enrichment exists
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:
    import config as C

WORKLIST_PATH = os.path.join(C.EXTERNAL_DIR, "newcomers_to_enrich.csv")
ENRICHED_PATH = os.path.join(C.EXTERNAL_DIR, "newcomers_enriched.csv")
PRIORS_PATH = os.path.join(C.PROCESSED_DIR, "newcomer_priors.csv")

ENRICHED_SCHEMA = [
    "Id", "matched_name", "source", "prev_club", "prev_league", "age",
    "minutes_prev", "matches_prev", "goals_prev", "assists_prev", "xg_prev", "xa_prev",
    "rating_prev", "transfer_type", "transfer_fee_eur_m", "confidence", "notes",
]


def _load_master() -> pd.DataFrame:
    return pd.read_csv(os.path.join(C.PROCESSED_DIR, "players_master.csv"))


def detect(top_n: int | None = None) -> pd.DataFrame:
    """Return the newcomer worklist and (re)write newcomers_to_enrich.csv."""
    top_n = C.NEWCOMER_ENRICH_TOP_N if top_n is None else top_n
    master = _load_master()
    new = master[master["is_newcomer"] == 1].copy()
    new = new.sort_values("FVM", ascending=False).head(top_n)
    work = new[["Id", "Nome", "Squadra", "R", "RM", "Qt.A", "FVM"]].copy()
    # Add empty enrichment columns as a template to fill.
    for col in ENRICHED_SCHEMA:
        if col not in work.columns and col != "Id":
            work[col] = ""
    work.to_csv(WORKLIST_PATH, index=False, encoding="utf-8")
    print(f"[newcomers] {len(new)} newcomer (top {top_n} per FVM) -> {WORKLIST_PATH}")
    return new


def load_enriched() -> pd.DataFrame:
    if not os.path.exists(ENRICHED_PATH):
        print(f"[newcomers] nessun file arricchito in {ENRICHED_PATH} (uso fallback priors)")
        return pd.DataFrame(columns=ENRICHED_SCHEMA)
    df = pd.read_csv(ENRICHED_PATH)
    df["Id"] = pd.to_numeric(df["Id"], errors="coerce").astype("Int64")
    df = df[df["Id"].notna()].copy()
    df["Id"] = df["Id"].astype(int)
    for col in ("minutes_prev", "matches_prev", "goals_prev", "assists_prev",
                "xg_prev", "xa_prev", "rating_prev", "age", "transfer_fee_eur_m"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[newcomers] arricchiti caricati: {len(df)} righe da {ENRICHED_PATH}")
    return df


def _fallback_prior(row: pd.Series) -> dict:
    """Role/FVM-based prior when no web data exists. HIGH uncertainty by construction.

    Anchored to FVM (the market's own expectation) so it is not arbitrary, but the value
    is explicitly flagged low-confidence and its uncertainty is inflated downstream.
    """
    # Without web data we have no evidence the player will start, so assume a conservative,
    # rotation-level workload anchored to FVM (the market's own signal): higher FVM ⇒ more
    # expected minutes. Deliberately cautious so unknowns don't masquerade as value picks.
    fvm = float(row.get("FVM") or 0)
    exp_share = float(np.clip(0.18 + fvm / 120.0, 0.15, 0.60))
    return {
        "age": np.nan,
        "exp_minutes_share": exp_share,
        "goals90_prior": np.nan,
        "assists90_prior": np.nan,
        "rating_prior": np.nan,
        "league_strength": np.nan,
        "transfer_type": "",
        "prior_source": "fallback_fvm_role",
        "prior_confidence": "low",
    }


def _web_prior(row: pd.Series, e: pd.Series) -> dict:
    """Translate enriched prior-league output to a Serie A per-90 prior."""
    lg = str(e.get("prev_league", "")).strip() or "Other"
    strength = C.LEAGUE_STRENGTH.get(lg, C.LEAGUE_STRENGTH["Other"])
    minutes = e.get("minutes_prev", np.nan)
    goals = e.get("goals_prev", np.nan)
    assists = e.get("assists_prev", np.nan)
    xg = e.get("xg_prev", np.nan)
    xa = e.get("xa_prev", np.nan)
    per90 = (minutes / 90.0) if pd.notna(minutes) and minutes > 0 else np.nan
    # Prefer xG/xA when available (more stable), else actual goals/assists.
    g_src = xg if pd.notna(xg) else goals
    a_src = xa if pd.notna(xa) else assists
    goals90 = (g_src / per90 * strength) if pd.notna(g_src) and pd.notna(per90) else np.nan
    assists90 = (a_src / per90 * strength) if pd.notna(a_src) and pd.notna(per90) else np.nan
    # Expected minutes share: prior workload discounted for adaptation AND for the quality gap
    # of the source league (a dominant Serie B/weak-league starter won't necessarily start in
    # Serie A). league_factor in [0.5, 1.0]; Serie A-equivalent leagues get no minutes penalty.
    league_factor = float(np.clip(strength, 0.5, 1.0))
    exp_share = np.nan
    if pd.notna(minutes):
        exp_share = (min(1.0, minutes / (C.MAX_MATCHDAYS * 90.0))
                     * C.NEWCOMER_MINUTES_DISCOUNT * league_factor)
    conf = str(e.get("confidence", "")).strip().lower() or "med"
    return {
        "age": e.get("age", np.nan),
        "exp_minutes_share": exp_share if pd.notna(exp_share) else C.NEWCOMER_MINUTES_DISCOUNT * 0.65,
        "goals90_prior": goals90,
        "assists90_prior": assists90,
        "rating_prior": e.get("rating_prev", np.nan),
        "league_strength": strength,
        "transfer_type": str(e.get("transfer_type", "")).strip(),
        "prior_source": f"web:{e.get('source','')}|{lg}",
        "prior_confidence": conf if conf in ("high", "med", "low") else "med",
    }


def build_priors() -> str:
    master = _load_master()
    new = master[master["is_newcomer"] == 1].copy()
    enriched = load_enriched().set_index("Id") if not load_enriched().empty else pd.DataFrame()

    rows = []
    for _, r in new.iterrows():
        if not enriched.empty and r["Id"] in enriched.index:
            prior = _web_prior(r, enriched.loc[r["Id"]])
        else:
            prior = _fallback_prior(r)
        prior.update({"Id": int(r["Id"]), "Nome": r["Nome"], "Squadra": r["Squadra"],
                      "primary_mantra": r["primary_mantra"], "FVM": r["FVM"]})
        rows.append(prior)
    out = pd.DataFrame(rows)
    out.to_csv(PRIORS_PATH, index=False, encoding="utf-8")
    n_web = (out["prior_source"].str.startswith("web")).sum() if len(out) else 0
    print(f"[newcomers] priors: {len(out)} newcomer ({n_web} da web, "
          f"{len(out)-n_web} fallback) -> {PRIORS_PATH}")
    return PRIORS_PATH


if __name__ == "__main__":
    if "--priors" in sys.argv:
        build_priors()
    else:
        detect()
