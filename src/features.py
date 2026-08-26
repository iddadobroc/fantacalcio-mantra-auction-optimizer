"""
features.py — Phase 3 feature engineering (leak-free, temporal).

For a given TARGET season, features are built ONLY from seasons strictly before it
(lag-1 = previous season, lag-2 = two seasons before), plus static attributes (role,
team strength from the prior season, newcomer priors, age). Target-season stats are NEVER
used as predictors → no temporal leakage.

Modeling tables produced (via build_modeling_table):
  - target 2024-25 : train fold  (features from 2023-24)
  - target 2025-26 : OOS test fold (features from 2024-25 [+2023-24])
  - target 2026-27 : prediction set (features from 2025-26 [+2024-25]); targets are NaN

Targets (when the season is historical):
  - target_pv    : appearances with a vote (minutes proxy)
  - target_fm    : fantamedia over played matches
  - target_total : Pv * Fm  = total seasonal fantasy production (the value target)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

try:
    from . import config as C
    from . import dataio
except ImportError:
    import config as C
    import dataio

LAG_STATS = ["Pv", "Mv", "Fm", "Gf", "Ass", "Amm", "Esp", "Rplus", "Gs"]
LAG_RATES = ["Gf_per_pv", "Ass_per_pv", "Amm_per_pv", "Esp_per_pv", "Rplus_per_pv"]


def _prev_seasons(target: str) -> tuple[str | None, str | None]:
    i = C.SEASON_ORDER.index(target)
    lag1 = C.SEASON_ORDER[i - 1] if i - 1 >= 0 else None
    lag2 = C.SEASON_ORDER[i - 2] if i - 2 >= 0 else None
    return lag1, lag2


def team_season_strength(stats_long: pd.DataFrame) -> dict:
    """(team, season) -> {attack, defense, avg_fm} from played rows, Pv-weighted where sensible."""
    out = {}
    played = stats_long[stats_long["played"] == 1]
    for (team, season), g in played.groupby(["Squadra", "season"]):
        w = g["Pv"].clip(lower=1)
        avg_fm = np.average(g["Fm"], weights=w) if len(g) else np.nan
        # Attack: total goals scored by the team's players (per player-season, summed).
        attack = g["Gf"].sum()
        # Defense: goals conceded, taken from goalkeepers' Gs (best available proxy).
        gk = g[g["Rm"] == "Por"]
        defense = gk["Gs"].sum() if len(gk) else np.nan
        out[(team, season)] = {"attack": attack, "defense": defense, "avg_fm": avg_fm}
    return out


def prior_team_strength(team: str, target: str, strength: dict) -> dict:
    """Most recent season strictly before `target` with data for `team`."""
    idx = C.SEASON_ORDER.index(target)
    for j in range(idx - 1, -1, -1):
        s = C.SEASON_ORDER[j]
        if (team, s) in strength:
            d = dict(strength[(team, s)])
            d["promoted"] = 0
            d["from_season"] = s
            return d
    # No prior Serie A data for this team (promoted) -> flag + neutral/low defaults.
    return {"attack": np.nan, "defense": np.nan, "avg_fm": np.nan,
            "promoted": 1, "from_season": None}


def _lag_frame(stats_long: pd.DataFrame, season: str, suffix: str) -> pd.DataFrame:
    if season is None:
        return pd.DataFrame(columns=["Id"])
    d = stats_long[stats_long["season"] == season].copy()
    cols = ["Id"] + LAG_STATS + LAG_RATES + ["played"]
    d = d[cols].rename(columns={c: f"{c}_{suffix}" for c in cols if c != "Id"})
    return d


def _newcomer_priors() -> pd.DataFrame:
    p = os.path.join(C.PROCESSED_DIR, "newcomer_priors.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame(columns=["Id"])


def build_modeling_table(target: str, stats_long: pd.DataFrame,
                         master: pd.DataFrame, strength: dict) -> pd.DataFrame:
    lag1, lag2 = _prev_seasons(target)

    # ---- Base pool + role/team + targets ------------------------------------
    if target == C.TARGET_SEASON:
        base = master.rename(columns={"Squadra": "team"}).copy()
        base["role"] = base["primary_mantra"]
        base["macro"] = base["role"].map(C.ROLE_TO_MACRO)
        base["target_pv"] = np.nan
        base["target_fm"] = np.nan
        base["target_total"] = np.nan
    else:
        d = stats_long[stats_long["season"] == target].copy()
        d = d.rename(columns={"Squadra": "team", "primary_mantra": "role"})
        d["macro"] = d["role"].map(C.ROLE_TO_MACRO)
        d["target_pv"] = d["Pv"]
        d["target_fm"] = d["Fm"]
        d["target_total"] = d["Pv"] * d["Fm"]
        base = d[["Id", "Nome", "team", "role", "macro",
                  "target_pv", "target_fm", "target_total"]].copy()

    # ---- Lag features -------------------------------------------------------
    base = base.merge(_lag_frame(stats_long, lag1, "l1"), on="Id", how="left")
    base = base.merge(_lag_frame(stats_long, lag2, "l2"), on="Id", how="left")
    # Guarantee every expected lag column exists (the earliest fold has no lag-2).
    for suffix in ("l1", "l2"):
        for c in LAG_STATS + LAG_RATES + ["played"]:
            col = f"{c}_{suffix}"
            if col not in base.columns:
                base[col] = np.nan

    # ---- Trends & history depth --------------------------------------------
    base["trend_fm"] = base["Fm_l1"] - base["Fm_l2"]
    base["trend_pv"] = base["Pv_l1"] - base["Pv_l2"]
    base["n_hist"] = base[["played_l1", "played_l2"]].fillna(0).sum(axis=1)

    # ---- Team strength (prior season) --------------------------------------
    ts = base["team"].apply(lambda t: prior_team_strength(t, target, strength))
    base["team_attack"] = ts.apply(lambda d: d["attack"])
    base["team_defense"] = ts.apply(lambda d: d["defense"])
    base["team_avg_fm"] = ts.apply(lambda d: d["avg_fm"])
    base["team_promoted"] = ts.apply(lambda d: d["promoted"])

    # ---- Newcomer flags & priors -------------------------------------------
    if target == C.TARGET_SEASON:
        # is_newcomer / is_returning / sa_seasons already present (base came from master).
        pri = _newcomer_priors()
        if not pri.empty:
            base = base.merge(
                pri[["Id", "age", "exp_minutes_share", "goals90_prior", "assists90_prior",
                     "league_strength", "prior_confidence"]], on="Id", how="left")
    else:
        # Historical folds: newcomer status is relative to that season's own history.
        hist_before = (stats_long[(stats_long["played"] == 1)]
                       .assign(idx=lambda x: x["season"].map(C.SEASON_ORDER.index)))
        tgt_idx = C.SEASON_ORDER.index(target)
        seen = set(hist_before[hist_before["idx"] < tgt_idx]["Id"])
        base["is_newcomer"] = (~base["Id"].isin(seen)).astype(int)
        base["is_returning"] = 0
        base["sa_seasons"] = base["Id"].map(
            hist_before[hist_before["idx"] < tgt_idx].groupby("Id")["season"].nunique()).fillna(0)
    for col in ("age", "exp_minutes_share", "goals90_prior", "assists90_prior",
                "league_strength", "prior_confidence"):
        if col not in base.columns:
            base[col] = np.nan

    base["target_season"] = target
    return base


def build_all() -> dict:
    stats_long = pd.read_csv(os.path.join(C.PROCESSED_DIR, "stats_long.csv"))
    master = pd.read_csv(os.path.join(C.PROCESSED_DIR, "players_master.csv"))
    strength = team_season_strength(stats_long)
    tables = {}
    for target in C.LABELED_TARGETS + [C.TARGET_SEASON]:
        t = build_modeling_table(target, stats_long, master, strength)
        tables[target] = t
        out = os.path.join(C.PROCESSED_DIR, f"features_{target}.csv")
        t.to_csv(out, index=False, encoding="utf-8")
        print(f"[features] {target}: {len(t)} righe, {t.shape[1]} colonne -> {out}")
    return tables


if __name__ == "__main__":
    build_all()
