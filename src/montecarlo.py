"""
montecarlo.py — Phase 9: Monte Carlo auction simulation.

Simulates opponents' willingness-to-pay to answer the strategic questions:
  - which players are INDISPENSABLE (big value drop to the next alternative) vs SUBSTITUTABLE;
  - the best VALUE picks (high VAR per credit AND realistically gettable);
  - how much you can SAFELY spend on a top player without compromising the rest of the roster;
  - which players you must AVOID overpaying for (steep price tail).

Model (assumptions, all in config, all declared):
  Each opponent's bid for a player ~ exp_price * LogNormal(0, MC_OPPONENT_PRICE_NOISE) *
  MC_OPPONENT_AGGRESSION. The market clearing price = max opponent bid (you must beat it).
  This is a deliberately simple opponent model — it captures price dispersion, not tactics.

Outputs: outputs/montecarlo.csv (per player), outputs/auction_strategy.md (summary).
Run: python src/montecarlo.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:
    import config as C

REPLACEMENT_ROSTER = {"P": 3, "D": 8, "C": 8, "A": 6}


def simulate_prices(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """N_SIMS x N_players matrix of clearing prices = max over opponents' bids."""
    n = len(df)
    base = df["exp_price"].to_numpy(float)[None, :]
    n_opp = C.N_PARTICIPANTS - 1
    # opponent bids: base * lognormal, take the max across opponents per sim
    clearing = np.empty((C.MC_N_SIMS, n))
    for k in range(0, C.MC_N_SIMS, 500):                     # chunk to bound memory
        m = min(500, C.MC_N_SIMS - k)
        noise = rng.lognormal(0.0, C.MC_OPPONENT_PRICE_NOISE, size=(m, n_opp, n))
        bids = base[:, None, :] * noise * C.MC_OPPONENT_AGGRESSION
        clearing[k:k+m] = bids.max(axis=1)
    return np.maximum(1, np.round(clearing))


def indispensability(df: pd.DataFrame) -> pd.DataFrame:
    """Value drop to the next-best alternative in the same primary role."""
    df = df.copy()
    df["gap_to_next"] = 0.0
    df["alternatives"] = 0
    for role, g in df.groupby("role"):
        g = g.sort_values("exp_total", ascending=False)
        vals = g["exp_total"].to_numpy()
        nxt = np.append(vals[1:], vals[-1] if len(vals) else 0)
        df.loc[g.index, "gap_to_next"] = (vals - nxt).round(1)
        # alternatives = players within 15% of this player's value in the same role
        for idx, v in zip(g.index, vals):
            df.loc[idx, "alternatives"] = int((g["exp_total"] >= 0.85 * v).sum() - 1)
    return df


def run() -> str:
    rng = np.random.default_rng(C.RANDOM_STATE)
    df = pd.read_csv(os.path.join(C.PROCESSED_DIR, "player_values.csv"))
    df = df.sort_values("exp_total", ascending=False).reset_index(drop=True)

    prices = simulate_prices(df, rng)
    df["sim_price_median"] = np.median(prices, axis=0).astype(int)
    df["sim_price_p90"] = np.percentile(prices, 90, axis=0).astype(int)
    df["prob_bargain"] = (prices <= df["exp_price"].to_numpy()[None, :]).mean(axis=0).round(2)
    df["overpay_above"] = df["sim_price_p90"]                # paying beyond p90 = overpaying

    df = indispensability(df)
    # Indispensability score: big value gap + few alternatives, normalized 0-100 within macro.
    parts = []
    for macro, g in df.groupby("macro"):
        gg = g.copy()
        gap_n = gg["gap_to_next"] / max(1e-9, gg["gap_to_next"].max())
        alt_n = 1 - gg["alternatives"] / max(1, gg["alternatives"].max())
        gg["indispensability"] = (100 * (0.6 * gap_n + 0.4 * alt_n)).round(0)
        parts.append(gg)
    df = pd.concat(parts).sort_values("exp_total", ascending=False)
    df["verdict"] = np.where(df["indispensability"] >= 60, "Indispensabile",
                    np.where(df["indispensability"] >= 35, "Importante", "Sostituibile"))

    # Safe spend on a top player = BUDGET - (cost of a solid REST of the roster at par prices).
    # "par starter" = clearing price of the last starter-quality player leaguewide per macro
    # (rank = N_PARTICIPANTS * typical starters). Bench slots assumed cheap.
    STARTERS = {"P": 1, "D": 4, "C": 4, "A": 2}              # a typical XI
    BENCH_UNIT = 2                                            # avg credits per bench slot
    par_starter = {}
    for macro in REPLACEMENT_ROSTER:
        sub = df[df["macro"] == macro].sort_values("exp_total", ascending=False)
        rank = min(int(C.N_PARTICIPANTS * STARTERS[macro]), len(sub) - 1)
        par_starter[macro] = int(sub["sim_price_median"].iloc[rank]) if len(sub) else 1
    reserve_starters = sum(STARTERS[m] * par_starter[m] for m in STARTERS)
    bench_slots = sum(REPLACEMENT_ROSTER[m] - STARTERS[m] for m in STARTERS)
    bench_cost = bench_slots * BENCH_UNIT

    def safe_spend(row):
        # pay for THIS player + the other 10 starters at par + cheap bench
        rest_starters = reserve_starters - par_starter[row["macro"]]
        return int(max(1, C.BUDGET - rest_starters - bench_cost))
    df["safe_max_spend"] = df.apply(safe_spend, axis=1)
    par = par_starter

    out_cols = ["Id", "Nome", "team", "role", "macro", "exp_total", "VAR", "exp_price",
                "sim_price_median", "sim_price_p90", "prob_bargain", "overpay_above",
                "gap_to_next", "alternatives", "indispensability", "verdict",
                "safe_max_spend", "value_per_price", "risk", "is_newcomer"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].round(3).to_csv(os.path.join(C.PROCESSED_DIR, "montecarlo.csv"),
                                 index=False, encoding="utf-8")

    # ---- strategy summary ----
    md = ["# Strategia d'asta — simulazione Monte Carlo\n",
          f"_{C.MC_N_SIMS} simulazioni; {C.N_PARTICIPANTS} partecipanti; rumore prezzi "
          f"σ={C.MC_OPPONENT_PRICE_NOISE}, aggressività={C.MC_OPPONENT_AGGRESSION}. "
          "Modello avversari volutamente semplice (dispersione prezzi, non tattica)._\n"]

    md.append("## Giocatori INDISPENSABILI (forte calo verso l'alternativa)\n")
    ind = df[df["verdict"] == "Indispensabile"].sort_values("exp_total", ascending=False).head(15)
    md.append(ind[["Nome", "team", "role", "exp_total", "gap_to_next", "alternatives",
                   "sim_price_median", "safe_max_spend"]].to_markdown(index=False) + "\n")

    md.append("## Migliori VALUE PICK per reparto (alto VAR/credito)\n")
    md.append("_Presentati per reparto: i portieri sono strutturalmente efficienti (giochi 1 solo "
              "titolare), quindi non vanno confrontati con i giocatori di movimento._\n")
    for macro, label in [("P", "Portieri"), ("D", "Difensori"),
                         ("C", "Centrocampisti"), ("A", "Attaccanti")]:
        vp = df[(df["macro"] == macro) & (df["VAR"] > 0) & (df["exp_price"] >= 3)].sort_values(
            "value_per_price", ascending=False).head(6)
        md.append(f"**{label}**\n")
        md.append(vp[["Nome", "team", "role", "exp_total", "VAR", "exp_price",
                      "sim_price_median", "prob_bargain"]].to_markdown(index=False) + "\n")

    md.append("## Da NON sovrapagare (coda di prezzo ripida — verdetto Sostituibile ma costoso)\n")
    avoid = df[(df["verdict"] == "Sostituibile") & (df["exp_price"] >= 15)].sort_values(
        "exp_price", ascending=False).head(12)
    md.append(avoid[["Nome", "team", "role", "exp_price", "sim_price_p90", "alternatives",
                     "verdict"]].to_markdown(index=False) + "\n")

    md.append("## Spesa sicura sui top (senza compromettere la rosa)\n")
    md.append("`safe_max_spend` = crediti massimi su quel giocatore riempiendo gli altri slot a "
              "prezzi 'par' simulati.\n")
    tops = df.sort_values("exp_total", ascending=False).head(12)
    md.append(tops[["Nome", "team", "role", "exp_price", "sim_price_median",
                    "safe_max_spend", "indispensability"]].to_markdown(index=False) + "\n")

    with open(os.path.join(C.OUTPUTS_DIR, "auction_strategy.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"[montecarlo] {C.MC_N_SIMS} sim -> data/processed/montecarlo.csv, outputs/auction_strategy.md")
    print(f"[montecarlo] par prices per macro: {par}")
    return os.path.join(C.PROCESSED_DIR, "montecarlo.csv")


if __name__ == "__main__":
    run()
