"""
preprocess.py — Phase 1 preprocessing.

Builds two clean, documented artifacts (CSV, in data/processed/):

  1. stats_long.csv      — one row per (Id, season) for the 3 historical seasons, cleaned,
                           with a `played` flag (Pv>0) so never-played rows are not treated
                           as real observations.
  2. players_master.csv  — the 2026/27 auction pool (one row per Id) with Mantra roles,
                           quotazioni/FVM, macro role, current team, and — importantly —
                           `sa_seasons` = number of prior Serie A seasons with real minutes.
                           `sa_seasons == 0` ⇒ genuine newcomer to the Serie A pool (to be
                           web-enriched). Players from promoted teams with older data are
                           NOT mislabelled as foreign newcomers.

No values are imputed here; missing stays missing (and is flagged).

Run:  python src/preprocess.py
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


def build_stats_long() -> pd.DataFrame:
    df = dataio.load_all_stats()
    # Drop rows without a usable Id.
    df = df[df["Id"].notna()].copy()
    df["Id"] = df["Id"].astype(int)
    # `played`: a real, usable observation only if the player got at least one vote.
    df["played"] = (df["Pv"].fillna(0) > 0).astype(int)
    # Mantra roles as a normalized ';'-joined string + primary role.
    df["mantra_roles"] = df["Rm"].apply(lambda x: ";".join(dataio.parse_mantra_roles(x)))
    df["primary_mantra"] = df["mantra_roles"].apply(lambda s: s.split(";")[0] if s else "")
    # Per-appearance rates (guard divide-by-zero; NaN where not played).
    with np.errstate(divide="ignore", invalid="ignore"):
        for stat in ("Gf", "Ass", "Amm", "Esp", "Rplus", "Rminus", "Au"):
            df[f"{stat}_per_pv"] = np.where(df["Pv"] > 0, df[stat] / df["Pv"], np.nan)
    cols = (["Id", "season", "R", "Rm", "mantra_roles", "primary_mantra", "Nome", "Squadra",
             "played", "Pv", "Mv", "Fm", "Gf", "Gs", "Rp", "Rc", "Rplus", "Rminus",
             "Ass", "Amm", "Esp", "Au"]
            + [f"{s}_per_pv" for s in ("Gf", "Ass", "Amm", "Esp", "Rplus", "Rminus", "Au")])
    return df[cols]


def build_players_master(stats_long: pd.DataFrame) -> pd.DataFrame:
    q = dataio.load_quotes()
    q = q[q["Id"].notna()].copy()
    q["Id"] = q["Id"].astype(int)
    # Exclude 'Ceduti' (already a separate sheet; the 'Tutti' sheet is the active pool).
    ceduti_ids = set()
    ced = dataio.load_ceduti()
    if not ced.empty and "Id" in ced.columns:
        ceduti_ids = set(pd.to_numeric(ced["Id"], errors="coerce").dropna().astype(int))
    q = q[~q["Id"].isin(ceduti_ids)].copy()

    q["mantra_roles"] = q["RM"].apply(lambda x: ";".join(dataio.parse_mantra_roles(x)))
    q["primary_mantra"] = q["mantra_roles"].apply(lambda s: s.split(";")[0] if s else "")

    # Serie A history depth: number of seasons with real minutes (played==1).
    hist = (stats_long[stats_long["played"] == 1]
            .groupby("Id")["season"].nunique().rename("sa_seasons"))
    last_seen = (stats_long[stats_long["played"] == 1]
                 .groupby("Id")["season"].max().rename("sa_last_season"))
    q = q.merge(hist, on="Id", how="left").merge(last_seen, on="Id", how="left")
    q["sa_seasons"] = q["sa_seasons"].fillna(0).astype(int)
    q["is_newcomer"] = (q["sa_seasons"] == 0).astype(int)     # genuine unknown to Serie A
    q["is_returning"] = ((q["sa_seasons"] > 0) &
                         (q["sa_last_season"] != "2025-26")).astype(int)  # older data only

    keep = ["Id", "Nome", "Squadra", "R", "RM", "mantra_roles", "primary_mantra",
            "Qt.A", "Qt.I", "Qt.A M", "Qt.I M", "FVM", "FVM M",
            "sa_seasons", "sa_last_season", "is_newcomer", "is_returning"]
    return q[keep].sort_values("FVM", ascending=False).reset_index(drop=True)


def run() -> tuple[str, str]:
    stats_long = build_stats_long()
    master = build_players_master(stats_long)

    p1 = os.path.join(C.PROCESSED_DIR, "stats_long.csv")
    p2 = os.path.join(C.PROCESSED_DIR, "players_master.csv")
    stats_long.to_csv(p1, index=False, encoding="utf-8")
    master.to_csv(p2, index=False, encoding="utf-8")

    n_new = int(master["is_newcomer"].sum())
    n_ret = int(master["is_returning"].sum())
    print(f"[preprocess] stats_long: {len(stats_long)} righe -> {p1}")
    print(f"[preprocess] players_master: {len(master)} giocatori -> {p2}")
    print(f"[preprocess] newcomer genuini (0 stagioni Serie A): {n_new}")
    print(f"[preprocess] rientri/dati vecchi (storia SA ma non 25/26): {n_ret}")
    print(f"[preprocess] con storia 25/26: {len(master) - n_new - n_ret}")
    return p1, p2


if __name__ == "__main__":
    run()
