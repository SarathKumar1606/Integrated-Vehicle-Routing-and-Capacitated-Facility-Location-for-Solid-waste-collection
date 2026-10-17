"""
Demo: re-plan one day after a waste surge, with exact DP routing.

Scenario: on the Chennai free-flow plan (the best saved solution), an event
at CEG Main Canteen makes its bin fill early -- it is full on a day it was not
scheduled for collection. dp_router.replan_day() adds it to that day's
collection and re-routes the day exactly; everything else stays as planned.

Run:  python scripts/demo_replan.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                        # noqa: E402

from decoder import decode                                # noqa: E402
from dp_router import replan_day, warm_up                 # noqa: E402
from loader import DAY_NAMES, load_instance               # noqa: E402
from optimizers import penalty_weights                    # noqa: E402

INSTANCE = "chennai_guindy"
SURGE_POINT = "CEG Main Canteen, CEG Square"


def best_solution(inst):
    best = None
    for f in sorted((ROOT / "results" / "solutions").glob(f"{INSTANCE}_[sg]a_seed*.npz")):
        with np.load(f) as z:
            sol = decode(inst, z["pop"], z["mask"], *penalty_weights(inst))
        if best is None or (not sol.feasible, sol.fitness) < best[0]:
            best = ((not sol.feasible, sol.fitness), f.name, sol)
    return best[1], best[2]


def main() -> None:
    inst = load_instance(ROOT / "data" / INSTANCE)
    with open(ROOT / "data" / INSTANCE / "points.csv", encoding="utf-8") as fh:
        names = {int(r["id"]) - 1: r["name"] for r in csv.DictReader(fh)
                 if int(r["id"]) > 0}
    p = next(i for i, n in names.items() if n == SURGE_POINT)
    src, sol = best_solution(inst)

    # the first working day on which the point is not collected
    day = next(d for d in inst.work_days
               if p not in [s for r in sol.routes_on(d) for s in r.stops])
    cap = float(inst.bin_cap[sol.bins[p]])
    print(f"Plan: {src}, overall {inst.money(sol.overall_cost)}/week")
    print(f"Surge: {names[p]} (bin {cap} m3) is full on {DAY_NAMES[day]}, "
          f"a day it was not scheduled. Its normal level that day: "
          f"{sol.w[p, day]:.2f} m3.\n")

    def show(routes):
        for k, r in enumerate(routes, 1):
            seq = " -> ".join(["depot"] + [names[s].split(",")[0] for s in r.stops]
                              + ["depot"])
            print(f"  R{k}  {r.total_load:5.2f} m3  {r.duration:6.2f} min  {seq}")

    print(f"{DAY_NAMES[day]} as planned:")
    show(sol.routes_on(day))

    warm_up()                                  # keep JIT start-up out of timing
    t0 = time.perf_counter()
    res = replan_day(inst, sol, day, {p: cap})
    first_ms = (time.perf_counter() - t0) * 1000
    reps = 200
    t0 = time.perf_counter()
    for _ in range(reps):
        replan_day(inst, sol, day, {p: cap})
    mean_ms = (time.perf_counter() - t0) * 1000 / reps

    print(f"\n{DAY_NAMES[day]} re-planned ({res.method}, "
          f"{'feasible' if res.feasible else 'INFEASIBLE'}):")
    show(res.routes)
    print(f"\n  route time   {res.old_minutes:.2f} -> {res.new_minutes:.2f} min "
          f"({res.new_minutes - res.old_minutes:+.2f})")
    print(f"  cost change  {inst.money(res.cost_change)} for {DAY_NAMES[day]} "
          f"(CCV {inst.money(inst.ccv)}/min)")
    print(f"  wall time    {first_ms:.2f} ms first call, {mean_ms:.2f} ms mean "
          f"over {reps} calls (DP itself: {res.ms:.2f} ms)")


if __name__ == "__main__":
    main()
