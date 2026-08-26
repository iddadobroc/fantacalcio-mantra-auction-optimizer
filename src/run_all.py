"""
run_all.py — run the full pipeline end-to-end, in order.

    python src/run_all.py

Steps: audit → preprocess → newcomer priors → model → value → optimize → montecarlo →
report → dashboard. Each step writes to data/processed and outputs/. Idempotent: safe to
re-run (e.g. after adding more season files, which are auto-discovered).
"""
from __future__ import annotations
import time

try:
    from . import audit, preprocess, newcomers, model, value, optimize, montecarlo, report, build_dashboard
except ImportError:
    import audit, preprocess, newcomers, model, value, optimize, montecarlo, report, build_dashboard


def main():
    t0 = time.time()
    steps = [
        ("Data audit", lambda: audit.run()),
        ("Preprocess", lambda: preprocess.run()),
        ("Newcomer priors", lambda: newcomers.build_priors()),
        ("Model + uncertainty", lambda: model.run()),
        ("Economic value", lambda: value.run()),
        ("Optimize + module selection", lambda: optimize.run()),
        ("Monte Carlo", lambda: montecarlo.run()),
        ("Report + Excel", lambda: report.run()),
        ("Auction dashboard", lambda: build_dashboard.build()),
    ]
    for i, (name, fn) in enumerate(steps, 1):
        print(f"\n{'='*70}\n[{i}/{len(steps)}] {name}\n{'='*70}")
        fn()
    print(f"\n✓ Pipeline completata in {time.time()-t0:.0f}s.")
    print("  Report:    outputs/report.md, outputs/player_rankings.xlsx")
    print("  Strategia: outputs/auction_strategy.md, outputs/module_comparison.csv")
    print("  DEMO ASTA: dashboard/auction_dashboard.html  (apri col doppio click)")


if __name__ == "__main__":
    main()
