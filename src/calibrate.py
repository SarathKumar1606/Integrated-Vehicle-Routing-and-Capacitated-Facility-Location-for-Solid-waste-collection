"""
Calibrate GA generations so GA and SA spend the same number of fitness
evaluations, as the paper's comparison does.

One SA run (seed 0) measures SA's evaluation count and wall time at a given
--iters-per-temp. GA spends pop_size evaluations on the initial population
and pop_size per generation, so the matching generation count is
round(SA_evals / pop_size) - 1. One GA run at that setting confirms it.

Usage:
    python src/calibrate.py --instance data/chennai_guindy --iters-per-temp 2500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from loader import load_instance
from optimizers import genetic_algorithm, simulated_annealing

POP_SIZE = 100          # genetic_algorithm's default population


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--iters-per-temp", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    folder = Path(args.instance)
    inst = load_instance(folder, args.name or folder.name)

    t0 = time.perf_counter()
    sa = simulated_annealing(inst, seed=args.seed,
                             iters_per_temp=args.iters_per_temp)
    t_sa = time.perf_counter() - t0

    generations = max(round(sa.evaluations / POP_SIZE) - 1, 1)
    t0 = time.perf_counter()
    ga = genetic_algorithm(inst, seed=args.seed, generations=generations)
    t_ga = time.perf_counter() - t0

    gap = (ga.evaluations - sa.evaluations) / sa.evaluations * 100
    print(f"instance {inst.name}")
    print(f"  SA  iters/temp {args.iters_per_temp:>6}  "
          f"temperatures {len(sa.history) - 1:>4}  evaluations {sa.evaluations:>8}  "
          f"wall {t_sa:7.1f} s  best {sa.best_fitness:.2f}")
    print(f"  GA  generations {generations:>5}  "
          f"evaluations {ga.evaluations:>8}  "
          f"wall {t_ga:7.1f} s  best {ga.best_fitness:.2f}")
    print(f"  GA vs SA evaluations: {gap:+.2f}%")
    print(f"CALIBRATED --generations {generations}  "
          f"SA_WALL {t_sa:.1f}  GA_WALL {t_ga:.1f}")


if __name__ == "__main__":
    main()
