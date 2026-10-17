"""
Polish the saved best solutions of existing SA/GA experiments with exact
per-day DP routing. SA and GA are NOT re-run: each seed's saved chromosome
(results/solutions/<name>_<algo>_seed<N>.npz) is decoded, checked against its
row in results/<name>_<algo>.csv, and polished.

Writes results/polish_<name>_<algo>.csv (one row per seed) and the polished
routes to results/solutions_polished/<name>_<algo>_seed<N>.npz.

Usage:
    python src/polish_experiment.py                      # the three instances
    python src/polish_experiment.py --instance data/12_1 --name i.12.1
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from decoder import decode
from dp_router import MAX_DP_POINTS, polish_with_report, routes_to_arrays, warm_up
from loader import load_instance
from optimizers import penalty_weights

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
DEFAULT = [("i.12.1", ROOT / "data" / "12_1"),
           ("chennai_guindy", ROOT / "data" / "chennai_guindy"),
           ("chennai_guindy_peak", ROOT / "data" / "chennai_guindy_peak")]


def polish_saved(name: str, folder: Path, algo: str) -> list[dict]:
    inst = load_instance(folder, name)
    lam, gamma = penalty_weights(inst)
    with (RESULTS / f"{name}_{algo}.csv").open() as fh:
        recorded = {int(r["seed"]): float(r["overall_cost"])
                    for r in csv.DictReader(fh)}
    out_dir = RESULTS / "solutions_polished"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in sorted(recorded):
        path = RESULTS / "solutions" / f"{name}_{algo}_seed{seed}.npz"
        with np.load(path) as z:
            pop, mask = z["pop"], z["mask"]
        sol = decode(inst, pop, mask, lam, gamma)
        if abs(round(sol.overall_cost, 4) - recorded[seed]) > 1e-6:
            raise RuntimeError(f"{path.name} decodes to {sol.overall_cost}, "
                               f"CSV says {recorded[seed]}")
        t0 = time.perf_counter()
        pol, rep = polish_with_report(inst, sol, lam, gamma)
        total_ms = (time.perf_counter() - t0) * 1000.0
        dp_days = [d for d in rep.days if d.status != "skipped_size"]
        np.savez(out_dir / path.name, pop=pop, mask=mask,
                 **routes_to_arrays(pol.routes))
        rows.append({
            "instance": name, "algorithm": algo.upper(), "seed": seed,
            "currency": inst.currency,
            "overall_cost": round(sol.overall_cost, 4),
            "routing_cost": round(sol.routing_cost, 4),
            "bin_cost": round(sol.bin_cost, 4),
            "polished_overall_cost": round(pol.overall_cost, 4),
            "polished_routing_cost": round(pol.routing_cost, 4),
            "routing_improvement": round(sol.routing_cost - pol.routing_cost, 4),
            "routing_improvement_pct": round(
                (sol.routing_cost - pol.routing_cost) / sol.routing_cost * 100, 4),
            "feasible": sol.feasible, "polished_feasible": pol.feasible,
            "routes_before": len(sol.routes), "routes_after": len(pol.routes),
            "days": len(rep.days),
            "days_improved": sum(d.status == "improved" for d in rep.days),
            "days_skipped": len(rep.skipped),
            "skipped_day_sizes": " ".join(str(d.n_points) for d in rep.skipped),
            "largest_dp_day": max((d.n_points for d in dp_days), default=0),
            "dp_ms_total": round(sum(d.dp_ms for d in dp_days), 3),
            "dp_ms_per_day": round(sum(d.dp_ms for d in dp_days)
                                   / max(len(dp_days), 1), 4),
            "polish_ms": round(total_ms, 3),
        })
    with (RESULTS / f"polish_{name}_{algo}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    targets = ([(args.name or Path(args.instance).name, Path(args.instance))]
               if args.instance else DEFAULT)
    warm_up()
    print(f"MAX_DP_POINTS = {MAX_DP_POINTS}")
    for name, folder in targets:
        inst = load_instance(folder, name)
        for algo in ("sa", "ga"):
            rows = polish_saved(name, folder, algo)
            imp = [r["routing_improvement"] for r in rows]
            changed = sum(r["days_improved"] > 0 for r in rows)
            skipped = sum(r["days_skipped"] for r in rows)
            best0 = min(r["overall_cost"] for r in rows)
            best1 = min(r["polished_overall_cost"] for r in rows)
            ms = np.mean([r["dp_ms_per_day"] for r in rows])
            print(f"  {name:<20} {algo.upper()}  mean routing gain "
                  f"{inst.money(float(np.mean(imp)))} "
                  f"({np.mean([r['routing_improvement_pct'] for r in rows]):.3f}%)  "
                  f"best {inst.money(best0)} -> {inst.money(best1)}  "
                  f"changed {changed}/{len(rows)}  DP {ms:.3f} ms/day  "
                  f"skipped days {skipped}")


if __name__ == "__main__":
    main()
