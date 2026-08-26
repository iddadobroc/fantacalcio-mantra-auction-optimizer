"""
dataio.py — Loading utilities for the raw Fantacalcio workbooks.

Handles the title-banner header row, standardises column names, coerces dtypes, and
attaches the season label. No cleaning decisions are made here beyond parsing; all
cleaning/merging logic lives in preprocess.py so it stays documented and auditable.
"""
from __future__ import annotations
import os
import pandas as pd

try:
    from . import config as C
except ImportError:  # allow running as a script
    import config as C

# Canonical column names for the stats workbooks.
STATS_COLS = ["Id", "R", "Rm", "Nome", "Squadra", "Pv", "Mv", "Fm",
              "Gf", "Gs", "Rp", "Rc", "Rplus", "Rminus", "Ass", "Amm", "Esp", "Au"]
STATS_RENAME = {"R+": "Rplus", "R-": "Rminus"}

NUMERIC_STATS = ["Pv", "Mv", "Fm", "Gf", "Gs", "Rp", "Rc",
                 "Rplus", "Rminus", "Ass", "Amm", "Esp", "Au"]
NUMERIC_QUOTES = ["Qt.A", "Qt.I", "Diff.", "Qt.A M", "Qt.I M", "Diff.M", "FVM", "FVM M"]


def load_stats_season(season: str) -> pd.DataFrame:
    """Load one stats workbook ('Tutti' sheet) and return a standardised DataFrame."""
    path = os.path.join(C.RAW_DIR, C.STATS_FILES[season])
    df = pd.read_excel(path, sheet_name="Tutti", header=C.EXCEL_HEADER_ROW)
    df = df.rename(columns=STATS_RENAME)
    df["season"] = season
    for col in NUMERIC_STATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("Id", "R", "Rm", "Nome", "Squadra"):
        if col == "Id":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = df[col].astype("string").str.strip()
    return df


def load_all_stats() -> pd.DataFrame:
    """Concatenate all available stats seasons into one long DataFrame."""
    return pd.concat([load_stats_season(s) for s in C.STATS_FILES], ignore_index=True)


def load_quotes(sheet: str = "Tutti") -> pd.DataFrame:
    """Load the 2026/27 quotazioni workbook (a given sheet)."""
    path = os.path.join(C.RAW_DIR, C.QUOTES_FILE)
    df = pd.read_excel(path, sheet_name=sheet, header=C.EXCEL_HEADER_ROW)
    df["season"] = C.TARGET_SEASON
    for col in NUMERIC_QUOTES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("Id", "R", "RM", "Nome", "Squadra"):
        if col == "Id":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = df[col].astype("string").str.strip()
    return df


def load_ceduti() -> pd.DataFrame:
    """Players marked as transferred out (Ceduti sheet) in the 2026/27 quotazioni."""
    try:
        return load_quotes(sheet="Ceduti")
    except Exception:
        return pd.DataFrame()


def parse_mantra_roles(rm: str | None) -> list[str]:
    """'Dd;Ds;E' -> ['Dd','Ds','E']. Returns [] for missing."""
    if rm is None or (isinstance(rm, float)) or str(rm).strip() in ("", "<NA>", "nan"):
        return []
    return [r.strip() for r in str(rm).split(";") if r.strip()]
