"""
optimize.py — Phase 6: roster optimization (ILP) + statistical module selection.

For each legal Mantra module we solve an Integer Linear Program that builds the best full
roster (25-30 players) within 500 credits, such that the module's starting XI can be fielded,
while also rewarding bench depth and penalising risk. Comparing the modules' solutions answers
the core question: **which module lets us build the best overall squad given the players and
prices available** — not merely which has the best 11 in isolation.

ILP (PuLP + CBC):
  variables  x[p] ∈{0,1}  player p in roster
             y[p,s]∈{0,1} player p is the STARTER in slot s (only for eligible (p,s))
  objective  Σ starter value + BENCH_VALUE_WEIGHT·Σ bench value − RISK_PENALTY·Σ risk·value
  s.t.       Σ price·x ≤ BUDGET
             ROSTER_MIN ≤ Σ x ≤ ROSTER_MAX
             each slot filled once: Σ_p y[p,s] = 1
             start ⇒ rostered: Σ_s y[p,s] ≤ x[p]
             macro bounds: min_m ≤ Σ_{p∈m} x ≤ max_m
             ≥1 bench GK (via macro P min)
             (auction reuse) forced-in players, removed 'taken' players, custom budget

Outputs: outputs/module_comparison.csv, outputs/optimal_roster.csv (+ per-module rosters).
Run: python src/optimize.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import pulp

try:
    from . import config as C
except ImportError:
    import config as C


def load_pool() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(C.PROCESSED_DIR, "player_values.csv"))
    df["mantra_set"] = df["mantra_roles"].fillna("").apply(
        lambda s: set(str(s).split(";")) - {""})
    # fall back to primary role if the mantra set is empty
    df.loc[df["mantra_set"].apply(len) == 0, "mantra_set"] = df["role"].apply(lambda r: {r})
    df["value"] = df["exp_total"].fillna(0).clip(lower=0)
    df["cost"] = df["exp_price"].fillna(1).clip(lower=1).astype(int)
    df["risk"] = df["risk"].fillna(df["risk"].median())
    return df


def _eligible(player_roles: set, slot_roles: frozenset) -> bool:
    return len(player_roles & set(slot_roles)) > 0


def solve_module(module: str, pool: pd.DataFrame, budget: int = None,
                 forced_ids: set = None, taken_ids: set = None,
                 roster_min: int = None, roster_max: int = None,
                 macro_have: dict = None, verbose: bool = False) -> dict:
    """Solve the ILP for one module. Returns dict with roster, starters, metrics, status.

    Auction-reuse params:
      forced_ids  : players already in MY roster (must be included, cost already paid → 0 here)
      taken_ids   : players removed from the pool (bought by others / me)
      budget      : remaining credits
      macro_have  : {macro: n_already_owned} to respect remaining macro capacity
    """
    budget = C.BUDGET if budget is None else budget
    roster_min = C.ROSTER_MIN if roster_min is None else roster_min
    roster_max = C.ROSTER_MAX if roster_max is None else roster_max
    forced_ids = forced_ids or set()
    taken_ids = (taken_ids or set()) - forced_ids
    macro_have = macro_have or {m: 0 for m in ("P", "D", "C", "A")}

    slots = C.MANTRA_MODULES[module]
    df = pool[~pool["Id"].isin(taken_ids)].copy().reset_index(drop=True)
    ids = df["Id"].tolist()
    val = dict(zip(ids, df["value"]))
    cost = dict(zip(ids, df["cost"]))
    risk = dict(zip(ids, df["risk"]))
    roles = dict(zip(ids, df["mantra_set"]))
    macro = dict(zip(ids, df["macro"]))
    forced = [i for i in ids if i in forced_ids]
    # forced players cost 0 additional (already paid); others cost their expected price
    eff_cost = {i: (0 if i in forced_ids else cost[i]) for i in ids}

    prob = pulp.LpProblem(f"roster_{module}", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in ids}
    # y[(i,s)] only for eligible pairs
    y = {}
    for s, slot in enumerate(slots):
        for i in ids:
            if _eligible(roles[i], slot):
                y[(i, s)] = pulp.LpVariable(f"y_{i}_{s}", cat="Binary")

    start_val = pulp.lpSum(val[i] * y[(i, s)] for (i, s) in y)
    bench_val = pulp.lpSum(val[i] * (x[i] - pulp.lpSum(
        y[(i, s)] for s in range(len(slots)) if (i, s) in y)) for i in ids)
    risk_pen = pulp.lpSum(risk[i] * val[i] * x[i] for i in ids)
    prob += start_val + C.BENCH_VALUE_WEIGHT * bench_val - C.RISK_PENALTY_WEIGHT * risk_pen

    # budget (only unpaid players consume remaining budget)
    prob += pulp.lpSum(eff_cost[i] * x[i] for i in ids) <= budget
    # roster size (counting forced already-owned players)
    prob += pulp.lpSum(x[i] for i in ids) >= roster_min
    prob += pulp.lpSum(x[i] for i in ids) <= roster_max
    # each slot filled exactly once
    for s in range(len(slots)):
        elig = [y[(i, s)] for i in ids if (i, s) in y]
        prob += pulp.lpSum(elig) == 1
    # start implies rostered
    for i in ids:
        ys = [y[(i, s)] for s in range(len(slots)) if (i, s) in y]
        if ys:
            prob += pulp.lpSum(ys) <= x[i]
    # MANTRA composition: only depth FLOORS driven by the reference module (no Classic quotas,
    # no per-macro maximum). GK is the sole hard positional floor; the XI slot constraints above
    # already guarantee the module is fieldable — these mins just ensure rotation depth.
    mm = C.module_macro_min(module)
    for m in ("P", "D", "C", "A"):
        sel = [x[i] for i in ids if macro[i] == m]
        have = macro_have.get(m, 0)
        if sel:
            prob += pulp.lpSum(sel) + have >= mm[m]
    # force-include owned players
    for i in forced:
        prob += x[i] == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return {"module": module, "status": status, "feasible": False}

    chosen = [i for i in ids if x[i].value() and x[i].value() > 0.5]
    starter_ids = {i for (i, s) in y if y[(i, s)].value() and y[(i, s)].value() > 0.5}
    slot_of = {i: s for (i, s) in y if y[(i, s)].value() and y[(i, s)].value() > 0.5}

    rec = df[df["Id"].isin(chosen)].copy()
    rec["is_starter"] = rec["Id"].isin(starter_ids).astype(int)
    rec["slot"] = rec["Id"].map(slot_of)
    rec["paid"] = rec["Id"].apply(lambda i: 0 if i in forced_ids else cost[i])
    rec["module"] = module

    starters = rec[rec["is_starter"] == 1]
    bench = rec[rec["is_starter"] == 0]
    metrics = {
        "module": module, "status": status, "feasible": True,
        "starter_points": round(starters["value"].sum(), 1),
        "bench_points": round(bench["value"].sum(), 1),
        "total_points": round(rec["value"].sum(), 1),
        "cost": int(rec["paid"].sum()),
        "roster_size": int(len(rec)),
        "avg_risk": round(float((rec["risk"] * rec["value"]).sum() / max(rec["value"].sum(), 1)), 3),
        "depth_ratio": round(bench["value"].sum() / max(starters["value"].sum(), 1), 3),
    }
    # Overall score: the ILP objective (starter + weighted bench − risk penalty).
    metrics["overall_score"] = round(
        metrics["starter_points"] + C.BENCH_VALUE_WEIGHT * metrics["bench_points"]
        - C.RISK_PENALTY_WEIGHT * (rec["risk"] * rec["value"]).sum(), 1)
    return {**metrics, "roster": rec}


def run() -> str:
    pool = load_pool()
    results, rosters = [], {}
    for module in C.MANTRA_MODULES:
        res = solve_module(module, pool)
        if res.get("feasible"):
            results.append({k: v for k, v in res.items() if k != "roster"})
            rosters[module] = res["roster"]
            print(f"[optimize] {module:8s} score={res['overall_score']:.0f} "
                  f"XI={res['starter_points']:.0f} cost={res['cost']} "
                  f"depth={res['depth_ratio']:.2f} risk={res['avg_risk']:.2f}")
        else:
            print(f"[optimize] {module:8s} INFEASIBLE ({res.get('status')})")

    comp = pd.DataFrame(results).sort_values("overall_score", ascending=False).reset_index(drop=True)
    comp.to_csv(os.path.join(C.OUTPUTS_DIR, "module_comparison.csv"), index=False, encoding="utf-8")

    best = comp.iloc[0]["module"]
    best_roster = rosters[best].sort_values(["is_starter", "value"], ascending=[False, False])
    cols = ["module", "Nome", "team", "role", "macro", "is_starter", "slot",
            "value", "exp_price", "paid", "VAR", "risk", "buyability"]
    cols = [c for c in cols if c in best_roster.columns]
    best_roster[cols].to_csv(os.path.join(C.OUTPUTS_DIR, "optimal_roster.csv"),
                             index=False, encoding="utf-8")
    # also persist every module's roster for the report / alternatives
    allr = pd.concat(rosters.values(), ignore_index=True)
    allr.to_csv(os.path.join(C.PROCESSED_DIR, "module_rosters.csv"), index=False, encoding="utf-8")

    print(f"\n[optimize] MODULO MIGLIORE: {best} "
          f"(score {comp.iloc[0]['overall_score']:.0f}, XI {comp.iloc[0]['starter_points']:.0f}, "
          f"cost {comp.iloc[0]['cost']})")
    print(f"[optimize] -> outputs/module_comparison.csv, outputs/optimal_roster.csv")
    return best


if __name__ == "__main__":
    run()
