"""
M6 - Experiment Runner
======================
Runs an algorithm many times with different seeds, in parallel across CPU
cores, and reports the same statistics the paper does: minimum, median, mean
and a 95% confidence interval. Every finished run is appended to a CSV
immediately, so a crash costs one run rather than the whole evening.

Usage:
    python src/runner.py --instance data/12_1 --name i.12.1 --algo sa --runs 10
    python src/runner.py --instance data/chennai_guindy --algo both --runs 10
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from decoder import decode
from loader import format_money, load_instance
from optimizers import genetic_algorithm, penalty_weights, simulated_annealing

ROOT = Path(__file__).parent.parent
SOLUTIONS_DIR = ROOT / "results" / "solutions"


def _one_run(args):
    folder, name, algo, seed, kwargs = args
    inst = load_instance(folder, name)
    t0 = time.time()
    res = (simulated_annealing(inst, seed=seed, **kwargs) if algo == "sa"
           else genetic_algorithm(inst, seed=seed, **kwargs))
    sol = decode(inst, res.best_pop, res.best_mask, *penalty_weights(inst))

    # Evaluations spent when each history entry was recorded. SA logs once
    # at the start and once per temperature step; GA logs once per
    # generation, starting after the initial population is scored.
    n_hist = len(res.history)
    if algo == "sa":
        history_evals = np.arange(n_hist) * (res.evaluations // max(n_hist - 1, 1))
    else:
        history_evals = (np.arange(n_hist) + 1) * (res.evaluations // n_hist)

    # keep the chromosome so exporter.py can rebuild and map the solution,
    # and the history for convergence plots
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(SOLUTIONS_DIR / f"{name}_{algo}_seed{seed}.npz",
             pop=res.best_pop, mask=res.best_mask, fitness=sol.fitness,
             history=np.asarray(res.history), history_evals=history_evals)
    return {
        "instance": name, "algorithm": res.algorithm, "seed": seed,
        "overall_cost": round(sol.overall_cost, 4),
        "bin_cost": round(sol.bin_cost, 4),
        "routing_cost": round(sol.routing_cost, 4),
        "currency": inst.currency,
        "feasible": sol.feasible,
        "evaluations": res.evaluations,
        "runtime_s": round(time.time() - t0, 2),
    }


def _read_done(out_csv: Path) -> list[dict]:
    """Rows already in a results CSV, with numbers and booleans parsed."""
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return []
    with out_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["evaluations"] = int(r["evaluations"])
        r["feasible"] = r["feasible"] == "True"
        for k in ("overall_cost", "bin_cost", "routing_cost", "runtime_s"):
            r[k] = float(r[k])
    return rows


def run_experiment(folder: str | Path, name: str, algo: str, runs: int = 10,
                   workers: int | None = None, out_csv: Path | None = None,
                   resume: bool = False, **kwargs) -> list[dict]:
    """Run seeds 0..runs-1. With resume=True, seeds that already have a row
    in the CSV and a saved solution are skipped and new rows are appended,
    so an interrupted experiment (crash, power loss) can be continued."""
    out_csv = out_csv or ROOT / "results" / f"{name}_{algo}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if resume:
        rows = [r for r in _read_done(out_csv)
                if (SOLUTIONS_DIR / f"{name}_{algo}_seed{r['seed']}.npz").exists()]
        if rows:
            print(f"  resuming: {len(rows)} seeds already done")
    done = {r["seed"] for r in rows}
    jobs = [(str(folder), name, algo, s, kwargs) for s in range(runs)
            if s not in done]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one_run, j) for j in jobs]
        with out_csv.open("w", newline="") as fh:
            writer = None
            if rows:                              # rewrite the kept rows
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
                fh.flush()
            for fut in as_completed(futures):
                row = fut.result()
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                fh.flush()                       # checkpoint after every run
                cost = format_money(row["overall_cost"], row["currency"])
                print(f"  seed {row['seed']:>2}  {row['algorithm']}  "
                      f"{cost:>14}  ({row['runtime_s']:.1f}s)")
    return sorted(rows, key=lambda r: r["seed"])


def summarize(rows: list[dict]) -> dict:
    """Min / median / mean / 95% CI, in the shape of the paper's Tables 8-9."""
    x = np.array([r["overall_cost"] for r in rows], dtype=float)
    n = len(x)
    mean, sd = float(x.mean()), float(x.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "min": float(x.min()),
        "median": float(np.median(x)),
        "mean": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "mean_runtime_s": float(np.mean([r["runtime_s"] for r in rows])),
        "all_feasible": all(r["feasible"] for r in rows),
    }


def print_summary(name: str, algo: str, s: dict, currency: str = "USD") -> None:
    m = lambda x: format_money(x, currency)  # noqa: E731
    print(f"\n{name}  [{algo.upper()}]  n={s['n']}  ({currency})")
    print(f"  min {m(s['min'])}   median {m(s['median'])}   "
          f"mean {m(s['mean'])}   95% CI [{m(s['ci_low'])}, {m(s['ci_high'])}]")
    print(f"  mean runtime {s['mean_runtime_s']:.1f} s   "
          f"all feasible: {s['all_feasible']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--algo", choices=["sa", "ga", "both"], default="both")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--iters-per-temp", type=int, default=1000)
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--resume", action="store_true",
                    help="skip seeds already in the results CSV and append")
    args = ap.parse_args()

    folder = Path(args.instance)
    name = args.name or folder.name

    for algo in (["sa", "ga"] if args.algo == "both" else [args.algo]):
        kw = ({"iters_per_temp": args.iters_per_temp} if algo == "sa"
              else {"generations": args.generations})
        print(f"\n=== {name}  {algo.upper()}  x{args.runs} ===")
        rows = run_experiment(folder, name, algo, args.runs,
                              workers=args.workers, resume=args.resume, **kw)
        print_summary(name, algo, summarize(rows),
                      rows[0].get("currency", "USD") if rows else "USD")


if __name__ == "__main__":
    main()
