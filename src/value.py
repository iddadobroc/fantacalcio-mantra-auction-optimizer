"""
value.py — Phase 5: economic value (VAR, expected auction price, max bid, buyability).

Turns fantasy-point projections (predictions.csv) into AUCTION economics:

  - Replacement level per macro role = value of the last player likely to be rostered
    league-wide (N_PARTICIPANTS * roster slots for that macro). VAR = exp_total - replacement.
  - Expected auction price: budget is split across macro roles by BUDGET_SHARE_BY_MACRO, then
    within a macro allocated proportionally to VAR^gamma, blended with FVM (the market's own
    signal). FVM is an INPUT, never taken as the price. Prices sum ≈ N_PARTICIPANTS * BUDGET.
  - Value/Price = VAR per expected credit.
  - Static max bid = fair value in credits * a small premium (willingness to pay for a target).
  - Buyability index (0-100): configurable blend of VAR, efficiency, low risk (static, full-budget
    view). The DYNAMIC, budget/slot-aware version lives in the auction engine + demo.

Output: data/processed/player_values.csv  (single source for optimizer, report, dashboard).
Run: python src/value.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:
    import config as C

# Replacement = the last player likely rostered league-wide per macro (roster depth). This
# gives a realistic auction price spread. NOTE: goalkeepers look credit-efficient on
# value/price because they score steadily for cheap — that is real, so "value picks" are
# always presented PER ROLE (you only start one GK); the optimizer caps roles anyway.
REPLACEMENT_ROSTER = {"P": 3, "D": 8, "C": 8, "A": 6}
# Convexity of price vs VAR. Calibrated to THIS league's actual auction (2025/26): very
# top-heavy ("stars & scrubs") — top players went 150-180, long tail at 1 credit.
PRICE_GAMMA = 1.55
PRICE_FVM_BLEND = 0.80      # lean on FVM (top-heavy market). Calibrato per 12 squadre (6000 cr totali,
                            # più concorrenza dell'asta 2025/26 a 10 squadre) → top più cari.
MAXBID_PREMIUM = 1.15       # static willingness-to-pay above fair value for a wanted player
SINGLE_PLAYER_CAP = 0.40    # a single player is never "worth" more than 40% of budget (static cap)

BUYABILITY_WEIGHTS = {"var": 0.40, "efficiency": 0.30, "low_risk": 0.30}

# Probable-lineups signal (data/external/roles_status.csv): expected appearances by projected
# role, blended with the model's estimate; plus penalty/set-piece season bonuses.
STATUS_PV = {"titolare": 32, "ballottaggio": 20, "riserva": 11}
STATUS_BLEND = 0.60          # weight on the projected role vs the model's historical estimate
# floor on expected appearances by projected role, so a weak model estimate (e.g. a returning
# player with no recent Serie A data) can't drag a projected STARTER too low.
STATUS_PV_FLOOR = {"titolare": 24, "ballottaggio": 12, "riserva": 0}
PEN_BONUS = {1: 6.0, 2: 2.0, 3: 1.0}    # season fantapoints for 1st/2nd/3rd penalty taker
SETPIECE_BONUS = 3.0         # season fantapoints (assist proxy) for set-piece takers
BALLOTTAGGIO_RISK = 0.10


def _load() -> pd.DataFrame:
    pred = pd.read_csv(os.path.join(C.PROCESSED_DIR, "predictions.csv"))
    master = pd.read_csv(os.path.join(C.PROCESSED_DIR, "players_master.csv"))
    df = pred.merge(master[["Id", "mantra_roles", "FVM", "Qt.A", "FVM M"]], on="Id", how="left")
    df["macro"] = df["macro"].fillna(df["role"].map(C.ROLE_TO_MACRO))
    df["exp_total"] = df["exp_total"].fillna(0).clip(lower=0)
    df = apply_status(df)       # projected role/penalties/set-pieces (current signal)
    df = apply_injuries(df)     # then discount for injuries
    return df


def _fuzzy_match(df, name, team, thr=82):
    """Best row in df matching name within team (fallback: whole df). Returns index or None."""
    try:
        from rapidfuzz import fuzz
        ratio = fuzz.ratio
    except Exception:
        def ratio(a, b):
            a, b = set(a), set(b)
            return 100 * len(a & b) / max(1, len(a | b))
    nm, tm = str(name).strip().lower(), str(team).strip().lower()
    cand = df[df["team"].astype(str).str.lower() == tm]
    if not len(cand):
        cand = df
    best, best_s = None, -1
    for idx, row in cand.iterrows():
        s = ratio(nm, str(row["Nome"]).strip().lower())
        if s > best_s:
            best_s, best = s, idx
    return best if best_s >= thr else None


def apply_status(df: pd.DataFrame) -> pd.DataFrame:
    """Fold in probable-lineups info: expected appearances by projected role (blended with the
    model), penalty-taker and set-piece season bonuses, and ballottaggio risk. Current, high-value
    signal for the auction. Configurable/declared; matched by team + fuzzy name."""
    df["starter_status"] = ""
    df["pen_rank"] = 0
    df["setpiece"] = 0
    path = os.path.join(C.EXTERNAL_DIR, "roles_status.csv")
    if not os.path.exists(path):
        return df
    st = pd.read_csv(path)
    st["status"] = st["status"].fillna("").astype(str).str.strip()  # empty status -> "" (not NaN)
    st["pen_rank"] = pd.to_numeric(st.get("pen_rank"), errors="coerce").fillna(0).astype(int)
    st["setpiece"] = pd.to_numeric(st.get("setpiece"), errors="coerce").fillna(0).astype(int)
    matched = 0
    for _, r in st.iterrows():
        idx = _fuzzy_match(df, r["name"], r["team"])
        if idx is None:
            continue
        matched += 1
        stt = r["status"]
        if stt and stt.lower() != "nan":
            df.loc[idx, "starter_status"] = stt
        df.loc[idx, "pen_rank"] = int(r["pen_rank"])
        df.loc[idx, "setpiece"] = int(r["setpiece"])
    # blend expected appearances toward the projected role, then recompute season total
    has = df["starter_status"] != ""
    if has.any() and "exp_fm" in df.columns:
        proj_pv = df["starter_status"].map(STATUS_PV)
        floor = df["starter_status"].map(STATUS_PV_FLOOR).fillna(0)
        blended = STATUS_BLEND * proj_pv + (1 - STATUS_BLEND) * df["exp_pv"]
        blended = blended.clip(lower=floor).clip(lower=0, upper=C.MAX_MATCHDAYS)
        df.loc[has, "exp_pv"] = blended[has]
        df.loc[has, "exp_total"] = (df["exp_pv"] * df["exp_fm"])[has]
    # penalty / set-piece season bonuses (added on top of the projected total)
    df["exp_total"] = df["exp_total"] + df["pen_rank"].map(PEN_BONUS).fillna(0) \
        + df["setpiece"] * SETPIECE_BONUS
    # ballottaggio -> more uncertainty
    df["risk"] = (df["risk"].fillna(0) + (df["starter_status"] == "ballottaggio") * BALLOTTAGGIO_RISK).clip(0, 1)
    print(f"[value] probabili formazioni applicate: {matched}/{len(st)} righe abbinate")
    return df


def apply_injuries(df: pd.DataFrame) -> pd.DataFrame:
    """Discount seasonal production by games missed (data/external/injuries.csv).

    A player expected out N of MAX_MATCHDAYS games loses ~N/38 of his total production, so
    exp_total/exp_pv/exp_goals/exp_assists are scaled by (38-N)/38. This flows into VAR, prices
    and recommendations. Risk is bumped for meaningful injuries. Minor knocks (N=1) are ~0 impact.
    Matching is by team + fuzzy name. New/absent injury files are handled gracefully.
    """
    df["games_out"] = 0
    df["injury_note"] = ""
    path = os.path.join(C.EXTERNAL_DIR, "injuries.csv")
    if not os.path.exists(path):
        return df
    inj = pd.read_csv(path)
    try:
        from rapidfuzz import fuzz
        def ratio(a, b): return fuzz.ratio(a, b)
    except Exception:
        def ratio(a, b):
            a, b = set(a), set(b)
            return 100 * len(a & b) / max(1, len(a | b))

    def norm(s): return str(s).strip().lower()
    matched = 0
    for _, r in inj.iterrows():
        nm, tm, gout = norm(r["name"]), norm(r["team"]), int(r["games_out"])
        cand = df[df["team"].astype(str).str.lower() == tm]
        if not len(cand):
            cand = df
        best, best_s = None, -1
        for idx, row in cand.iterrows():
            s = ratio(nm, norm(row["Nome"]))
            if s > best_s:
                best_s, best = s, idx
        if best is not None and best_s >= 82:
            df.loc[best, "games_out"] = gout
            df.loc[best, "injury_note"] = str(r["note"])
            matched += 1
    # apply availability factor
    fac = ((C.MAX_MATCHDAYS - df["games_out"]).clip(lower=0) / C.MAX_MATCHDAYS)
    for col in ("exp_total", "exp_pv", "exp_goals", "exp_assists"):
        if col in df.columns:
            df[col] = df[col] * fac
    # bump risk for real injuries (cap contribution)
    df["risk"] = (df["risk"].fillna(0) + (df["games_out"] / C.MAX_MATCHDAYS) * 0.7).clip(0, 1)
    print(f"[value] infortuni applicati: {matched}/{len(inj)} giocatori abbinati")
    return df


def compute_var(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["replacement"] = np.nan
    for macro, roster_n in REPLACEMENT_ROSTER.items():
        m = df["macro"] == macro
        sub = df[m].sort_values("exp_total", ascending=False)
        rank = int(round(C.N_PARTICIPANTS * roster_n * C.REPLACEMENT_DEPTH_MULT))
        rank = min(rank, len(sub) - 1) if len(sub) else 0
        repl = sub["exp_total"].iloc[rank] if len(sub) else 0.0
        df.loc[m, "replacement"] = repl
    df["VAR"] = (df["exp_total"] - df["replacement"]).round(2)
    df["VAR_pos"] = df["VAR"].clip(lower=0)
    return df


def compute_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["exp_price"] = 1.0
    for macro, share in C.BUDGET_SHARE_BY_MACRO.items():
        m = df["macro"] == macro
        sub = df[m]
        if not len(sub):
            continue
        pool = C.LEAGUE_TOTAL_BUDGET * share
        n_draft = int(C.N_PARTICIPANTS * REPLACEMENT_ROSTER[macro])
        # Draft pool = top-N by exp_total in this macro.
        order = sub.sort_values("exp_total", ascending=False)
        draft_idx = order.index[:n_draft]
        w = df.loc[draft_idx, "VAR_pos"].pow(PRICE_GAMMA)
        w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(draft_idx), index=draft_idx)
        # VAR-based allocation with a 1-credit floor per drafted player.
        var_price = 1.0 + (pool - len(draft_idx)) * w
        # FVM-based allocation (scale FVM within macro to the same pool).
        fvm = df.loc[draft_idx, "FVM"].fillna(0).clip(lower=0)
        fvm_price = (fvm / fvm.sum() * pool) if fvm.sum() > 0 else var_price
        blended = (1 - PRICE_FVM_BLEND) * var_price + PRICE_FVM_BLEND * fvm_price
        df.loc[draft_idx, "exp_price"] = blended.clip(lower=1)
    df["exp_price"] = df["exp_price"].round(0).clip(lower=1).astype(int)
    return df


def compute_bids_and_buyability(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Credits-per-VAR exchange rate per macro (for fair value & static max bid).
    df["fair_value"] = df["exp_price"]
    df["value_per_price"] = (df["VAR_pos"] / df["exp_price"]).round(3)
    cap = int(C.BUDGET * SINGLE_PLAYER_CAP)
    df["max_bid_static"] = np.minimum(
        np.round(df["exp_price"] * MAXBID_PREMIUM), cap).astype(int)
    df["max_bid_static"] = df[["max_bid_static", "exp_price"]].max(axis=1).astype(int)

    # Buyability 0-100 (static, full-budget view). Normalize per macro so each role is comparable.
    parts = []
    for macro in df["macro"].dropna().unique():
        m = df["macro"] == macro
        sub = df[m].copy()
        def _n(s):
            s = s.astype(float)
            rng = s.max() - s.min()
            return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)
        var_n = _n(sub["VAR_pos"])
        eff_n = _n(sub["value_per_price"])
        risk_n = 1 - sub["risk"].fillna(sub["risk"].median()).clip(0, 1)
        w = BUYABILITY_WEIGHTS
        score = 100 * (w["var"] * var_n + w["efficiency"] * eff_n + w["low_risk"] * risk_n)
        sub["buyability"] = score.round(1)
        parts.append(sub)
    df = pd.concat(parts).sort_values("exp_total", ascending=False)

    def _label(s):
        if s >= 75: return "Priorità alta"
        if s >= 55: return "Buon valore"
        if s >= 40: return "Prezzo pieno"
        if s >= 25: return "Sopravvalutato"
        return "Evita"
    df["buyability_label"] = df["buyability"].apply(_label)
    return df


def run() -> str:
    df = _load()
    df = compute_var(df)
    df = compute_prices(df)
    df = compute_bids_and_buyability(df)

    cols = ["Id", "Nome", "team", "role", "macro", "mantra_roles",
            "exp_pv", "exp_fm", "exp_total", "exp_goals", "exp_assists",
            "total_p10", "total_p50", "total_p90", "risk",
            "FVM", "Qt.A", "replacement", "VAR", "VAR_pos", "exp_price",
            "value_per_price", "max_bid_static", "buyability", "buyability_label",
            "is_newcomer", "pred_source", "prior_confidence", "games_out", "injury_note",
            "starter_status", "pen_rank", "setpiece"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].sort_values("exp_total", ascending=False).reset_index(drop=True)
    path = os.path.join(C.PROCESSED_DIR, "player_values.csv")
    out.round(3).to_csv(path, index=False, encoding="utf-8")

    spent = out["exp_price"].sum()
    print(f"[value] {len(out)} giocatori -> {path}")
    print(f"[value] somma prezzi attesi: {spent} (budget lega teorico {C.LEAGUE_TOTAL_BUDGET})")
    print(f"[value] prezzo atteso: min {out.exp_price.min()}, mediana {int(out.exp_price.median())}, max {out.exp_price.max()}")
    return path


if __name__ == "__main__":
    run()
