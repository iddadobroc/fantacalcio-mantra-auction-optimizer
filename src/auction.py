"""
auction.py — Phase 7: dynamic auction assistant (Python CLI, EXACT ILP reference).

Given what you've already bought (and who opponents took), it re-solves the roster ILP over
ALL modules on the REMAINING budget/slots and reports:
  - remaining budget & squad composition
  - the best module now, given what you already own
  - the recommended remaining roster (exact ILP)
  - top priority targets with a dynamic max bid
  - still-uncovered macro roles / alternatives

This is the exact reference behind the HTML demo (which uses a fast heuristic).

State: a JSON file (default data/processed/auction_state.json), e.g.
  {"mine": [{"name": "Barella", "price": 45}], "others": ["Douvikas"]}
Names are matched case-insensitively (fall back to substring). Ids also accepted.

Run:
  python src/auction.py                      # uses default state file (or empty)
  python src/auction.py --state path.json
  python src/auction.py --buy "Barella=45" --buy "Dimarco=30" --other "Douvikas"
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import pandas as pd

try:
    from . import config as C
    from . import optimize as OPT
except ImportError:
    import config as C
    import optimize as OPT

STATE_PATH = os.path.join(C.PROCESSED_DIR, "auction_state.json")


def _resolve(pool: pd.DataFrame, token) -> int | None:
    if isinstance(token, (int, float)) and not isinstance(token, bool):
        return int(token)
    s = str(token).strip().lower()
    if s.isdigit():
        return int(s)
    exact = pool[pool["Nome"].str.lower() == s]
    if len(exact):
        return int(exact.iloc[0]["Id"])
    sub = pool[pool["Nome"].str.lower().str.contains(s, regex=False)]
    if len(sub):
        return int(sub.sort_values("exp_total", ascending=False).iloc[0]["Id"])
    print(f"[auction] ATTENZIONE: giocatore non trovato: '{token}'")
    return None


def dyn_max_bid(row, budget_left: int, open_total: int, open_macro: dict,
                pool_avail: pd.DataFrame) -> int:
    """Same budget-pressure heuristic as the HTML demo, for consistency."""
    if open_total <= 0:
        return 0
    fair_remaining = max(1, C.BUDGET * open_total / C.ROSTER_MIN)
    pressure = max(0.4, min(2.5, budget_left / fair_remaining))
    macro_open = open_macro.get(row["macro"], 0)
    alt = len(pool_avail[(pool_avail["macro"] == row["macro"]) & (pool_avail["VAR"] > 0)])
    scarcity = (1 + 0.30 * max(0, 1 - alt / max(1, macro_open * 3))) if macro_open > 0 else 0.3
    spendable = max(1, budget_left - (open_total - 1))
    mb = round(row["exp_price"] * pressure * 1.15 * scarcity)
    if macro_open <= 0:
        mb = min(mb, round(spendable * 0.12))
    return int(max(1, min(mb, spendable)))


def analyze(state: dict) -> dict:
    pool = OPT.load_pool()
    mine = state.get("mine", [])
    others = state.get("others", [])
    mine_ids, paid = [], {}
    for item in mine:
        if isinstance(item, dict):
            pid = _resolve(pool, item.get("id", item.get("name")))
            pr = int(item.get("price", 0))
        else:
            pid, pr = _resolve(pool, item), 0
        if pid is not None:
            mine_ids.append(pid); paid[pid] = pr
    taken_ids = {i for i in (_resolve(pool, o) for o in others) if i is not None}
    forced = set(mine_ids)

    spent = sum(paid.values())
    budget_left = C.BUDGET - spent
    macro_have = {m: 0 for m in ("P", "D", "C", "A")}
    for pid in mine_ids:
        m = pool.loc[pool["Id"] == pid, "macro"]
        if len(m):
            macro_have[m.iloc[0]] = macro_have.get(m.iloc[0], 0) + 1

    # best module given current squad, on remaining budget
    results = {}
    for module in C.MANTRA_MODULES:
        res = OPT.solve_module(module, pool, budget=budget_left, forced_ids=forced,
                               taken_ids=taken_ids, macro_have=macro_have)
        if res.get("feasible"):
            results[module] = res
    if not results:
        return {"error": "Nessuna rosa fattibile coi vincoli/budget correnti.",
                "budget_left": budget_left, "spent": spent}
    best = max(results, key=lambda m: results[m]["overall_score"])
    roster = results[best]["roster"]

    # depth targets driven by the CURRENT best (reference) module — not Classic quotas
    ref_min = C.module_macro_min(best)
    open_macro = {m: max(0, ref_min[m] - macro_have.get(m, 0)) for m in ("P", "D", "C", "A")}
    open_total = max(C.ROSTER_MIN - len(mine_ids), sum(open_macro.values()))
    avail = pool[~pool["Id"].isin(taken_ids | forced)]

    targets = roster[~roster["Id"].isin(forced)].sort_values("value", ascending=False).copy()
    targets["max_bid"] = targets.apply(
        lambda r: dyn_max_bid(r, budget_left, open_total, open_macro, avail), axis=1)

    return {
        "spent": spent, "budget_left": budget_left,
        "owned": [pool.loc[pool.Id == i, "Nome"].iloc[0] for i in mine_ids],
        "macro_have": macro_have, "open_macro": open_macro, "open_total": open_total,
        "best_module": best, "module_score": round(results[best]["overall_score"], 1),
        "ref_min": ref_min,
        "targets": targets, "module_table": {m: round(r["overall_score"], 1) for m, r in results.items()},
    }


def print_report(a: dict):
    if a.get("error"):
        print(a["error"]); return
    print("=" * 64)
    print(f"  Crediti spesi: {a['spent']}   |   Residui: {a['budget_left']}")
    print(f"  Rosa attuale ({len(a['owned'])}/{C.ROSTER_MIN}-{C.ROSTER_MAX}): "
          f"{', '.join(a['owned']) or '—'}")
    print(f"  Composizione (libera; min per modulo rif. {a['best_module']}): " + "  ".join(
        f"{m} {a['macro_have'][m]}/{a['ref_min'][m]}" for m in ("P", "D", "C", "A")))
    print(f"  Slot ancora consigliati per il modulo: {a['open_total']}  "
          + "  ".join(f"{m}:{a['open_macro'][m]}" for m in a['open_macro'] if a['open_macro'][m] > 0))
    print("-" * 64)
    print(f"  MODULO MIGLIORE ORA: {a['best_module']}  (score {a['module_score']})")
    top3 = sorted(a["module_table"].items(), key=lambda x: -x[1])[:5]
    print("  Moduli (top 5 per score): " + ", ".join(f"{m}={s:.0f}" for m, s in top3))
    print("-" * 64)
    print("  PROSSIMI TARGET PRIORITARI (rosa ottimale residua):")
    cols = ["Nome", "team", "role", "macro", "value", "exp_price", "max_bid", "risk"]
    t = a["targets"].head(15)
    print(t[cols].to_string(index=False,
          formatters={"value": "{:.0f}".format, "risk": "{:.2f}".format}))
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=STATE_PATH)
    ap.add_argument("--buy", action="append", default=[], help='e.g. --buy "Barella=45"')
    ap.add_argument("--other", action="append", default=[], help='e.g. --other "Douvikas"')
    args = ap.parse_args()

    state = {"mine": [], "others": []}
    if os.path.exists(args.state):
        with open(args.state, encoding="utf-8") as f:
            state = json.load(f)
    for b in args.buy:
        name, _, price = b.partition("=")
        state["mine"].append({"name": name.strip(), "price": int(price or 0)})
    for o in args.other:
        state["others"].append(o.strip())

    print_report(analyze(state))


if __name__ == "__main__":
    main()
