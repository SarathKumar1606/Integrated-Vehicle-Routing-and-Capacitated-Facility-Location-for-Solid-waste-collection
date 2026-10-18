"""
Export ui/data/solver_core.json: everything a browser-side (JavaScript)
solver needs to score and search solutions by itself, plus known-answer and
parity fixtures to prove its decoder matches this one.

  instances   chennai_guindy and chennai_guindy_peak: travel-time matrix,
              waste, fleet, costs, penalty weights, bins, points
  validation  i.12.1 with the Appendix A chromosome and its expected answer
  parity      50 random (pop, mask) pairs per Chennai instance with the exact
              Python decode of each

Nothing is rounded: Python's json writes each float as its shortest repr,
which JSON.parse turns back into the identical IEEE-754 double.

Run:  python src/export_solver_core.py
Check: python scripts/verify_js_fixtures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import validate_appendix as va
from decoder import decode
from loader import DAY_NAMES, Instance, load_instance
from optimizers import penalty_weights, random_solution

ROOT = Path(__file__).parent.parent
OUT = ROOT / "ui" / "data" / "solver_core.json"
CHENNAI = ["chennai_guindy", "chennai_guindy_peak"]
PARITY_CASES = 50
PARITY_SEED = 20261018

DECODER_SPEC = {
    "_comment": (
        "How decoder.decode() turns (pop, mask) into costs, step by step and "
        "in the exact floating-point order needed to reproduce it bit for "
        "bit. Indices: points 0..nI-1; matrix C row/col 0 is the depot and "
        "point i is row/col i+1; days 0..n_days-1 (0 = MON); mask/pop "
        "columns are the working days in work_days order."),
    "chromosome": (
        "pop[i][c] = position of point i in working day c's visiting order "
        "(a permutation of 0..nI-1); mask[i][c] = 1 if point i is collected "
        "on working day c."),
    "1_accumulation": (
        "visited[i][t] = (mask[i][c] == 1) for t = work_days[c], false on rest "
        "days. carry = W (points only, W[1..nI]). Repeat up to 50 times: "
        "prev = copy(carry); for t in 0..n_days-1, for each i: "
        "w[i][t] = W[i] + carry[i]; carry[i] = visited[i][t] ? 0 : w[i][t]. "
        "Stop when for every i: carry[i] == prev[i] or "
        "|carry[i] - prev[i]| <= 1e-12 + 1e-5 * |prev[i]| (np.allclose "
        "defaults). wmax[i] = max over t of w[i][t]."),
    "2_bins": (
        "n_visits[i] = sum of mask[i][*] (as a float). For each bin b in "
        "order: fits = cap[b] >= wmax[i] - 1e-9; cost = bin_cost[b] + "
        "(ccv * service[b]) * n_visits[i] if fits else +Infinity. Choose the "
        "first b with the strictly smallest cost (ties -> lowest b). If the "
        "smallest is Infinity, the solution is invalid (no bin fits)."),
    "3_routes": (
        "For each working day in order: the points with mask 1, sorted by "
        "pop value ascending (stable: ties keep ascending point index). Walk "
        "them with load = 0: if the route is non-empty and "
        "load + w[i][day] > Q + 1e-9, close the route and start a new one "
        "(load = 0). Then load += w[i][day] and append i."),
    "4_duration": (
        "travel = C[0][s0+1], then travel += C[s_k+1][s_k+1+1] for each "
        "consecutive pair in order, then travel += C[s_last+1][0]. service = "
        "pairwise_sum([service[bins[s]] for s in route]). duration = "
        "(travel + service) + tu."),
    "pairwise_sum": (
        "NumPy's float64 sum of a[0..n-1]: result = 0.0 + block(a). For "
        "n < 8: r = -0.0; r += a[k] for k in order. For 8 <= n <= 128: "
        "r0..r7 = a[0..7]; for i = 8 step 8 while i < n - n%8: r0 += a[i], "
        "..., r7 += a[i+7]; res = ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)); then "
        "res += a[i] for the remaining i. For n > 128: n2 = floor(n/2) - "
        "(floor(n/2) % 8); block(a[0:n2]) + block(a[n2:n]) recursively."),
    "5_costs": (
        "bin_cost = pairwise_sum([bin_cost[bins[i]] for i in 0..nI-1]). "
        "total_minutes = 0.0; total_minutes += duration for each route in "
        "order (days ascending, routes in creation order). routing_cost = "
        "ccv * total_minutes. overall_cost = bin_cost + routing_cost."),
    "6_penalty": (
        "penalty = 0.0. For day in 0..n_days-1: if the day has "
        "n_routes > n_vehicles: penalty += lam * (n_routes - n_vehicles) / "
        "n_vehicles. Then for each route in order: if duration > TL + 1e-9: "
        "penalty += gamma * (duration - TL). feasible = (penalty == 0). "
        "fitness = overall_cost + penalty (what SA and GA minimise)."),
}


def _floats(a) -> list:
    return np.asarray(a, dtype=float).tolist()


def point_list(inst: Instance, folder: Path) -> list[dict]:
    names = {0: "Depot"}
    csv_path = folder / "points.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as fh:
            names.update({int(r["id"]): r["name"] for r in csv.DictReader(fh)})
    return [{"id": i, "name": names.get(i, f"Point {i}"),
             "lat": float(inst.lat[i]), "lon": float(inst.lon[i]),
             "is_depot": i == 0}
            for i in range(inst.n_points + 1)]


def instance_block(inst: Instance, folder: Path) -> dict:
    lam, gamma = penalty_weights(inst)
    return {
        "name": inst.name,
        "n_points": inst.n_points,
        "C": [_floats(row) for row in inst.C],
        "W": _floats(inst.W),
        "W_note": "W[0] is the depot (0); W[i] is point i-1's daily waste, m3",
        "Q": float(inst.Q),
        "n_vehicles": int(inst.n_vehicles),
        "TL": float(inst.TL),
        "ccv": float(inst.ccv),
        "tu": float(inst.tu),
        "n_days": int(inst.n_days),
        "day_names": list(DAY_NAMES[:inst.n_days]),
        "work_days": list(inst.work_days),
        "rest_days": list(inst.rest_days),
        "currency": inst.currency,
        "currency_symbol": inst.currency_symbol,
        "fx_rate_from_usd": inst.fx_rate,
        "cost_scale": float(inst.cost_scale),
        "lam": float(lam),
        "gamma": float(gamma),
        "bins": [{"combination": b, "capacity_m3": float(inst.bin_cap[b]),
                  "service_min": float(inst.bin_service[b]),
                  "weekly_cost": float(inst.bin_cost[b])}
                 for b in range(inst.n_bins)],
        "points": point_list(inst, folder),
    }


def solution_block(sol) -> dict:
    return {
        "bins": [int(b) for b in sol.bins],
        "bin_cost": float(sol.bin_cost),
        "routing_cost": float(sol.routing_cost),
        "overall_cost": float(sol.overall_cost),
        "penalty": float(sol.penalty),
        "fitness": float(sol.fitness),
        "feasible": bool(sol.feasible),
        "routes": [{"day": int(r.day), "stops": [int(s) for s in r.stops],
                    "duration": float(r.duration)} for r in sol.routes],
    }


def validation_block() -> dict:
    inst = load_instance(ROOT / "data" / "12_1", "i.12.1")
    lam, gamma = penalty_weights(inst)
    sol = decode(inst, va.POP, va.MASK, lam, gamma)
    block = instance_block(inst, ROOT / "data" / "12_1")
    block.pop("points")                       # Bahia Blanca: no names
    block.update({
        "pop": va.POP.tolist(),
        "mask": va.MASK.tolist(),
        "paper_expected": {
            "_comment": ("Gonzalez et al. (2025), Appendix A, Tables 16-17, "
                         "as published (2 decimals). Points are 1-based here, "
                         "as in the paper; days 0 = MON."),
            "bins": va.EXPECTED_BINS.tolist(),
            "w_work_days": va.EXPECTED_W.tolist(),
            "routes": [{"day": d, "stops_1based": s, "duration": t}
                       for d, s, t in va.EXPECTED_ROUTES],
            "tolerance": 5e-3,
        },
        "python_exact": solution_block(sol),
    })
    return block


def parity_cases(inst: Instance) -> list[dict]:
    lam, gamma = penalty_weights(inst)
    rng = np.random.default_rng(PARITY_SEED + inst.n_points + int(inst.TL))
    cases = []
    while len(cases) < PARITY_CASES:
        pop, mask = random_solution(inst, rng)
        try:
            sol = decode(inst, pop, mask, lam, gamma)
        except ValueError:                     # no bin fits: not a parity case
            continue
        cases.append({"pop": pop.tolist(), "mask": mask.astype(int).tolist(),
                      **solution_block(sol)})
    return cases


def feasible_cases(inst: Instance, name: str) -> list[dict]:
    """The saved best SA and GA solutions (all feasible), so the parity
    fixtures also cover the zero-penalty path that random schedules on these
    instances never reach."""
    lam, gamma = penalty_weights(inst)
    cases = []
    for algo in ("sa", "ga"):
        files = sorted((ROOT / "results" / "solutions").glob(f"{name}_{algo}_seed*.npz"),
                       key=lambda f: int(f.stem.rsplit("seed", 1)[1]))
        for f in files:
            with np.load(f) as z:
                pop, mask = z["pop"], z["mask"]
            sol = decode(inst, pop, mask, lam, gamma)
            cases.append({"source": f"{algo.upper()} seed {f.stem.rsplit('seed', 1)[1]}",
                          "pop": pop.tolist(), "mask": mask.astype(int).tolist(),
                          **solution_block(sol)})
    return cases


def main() -> None:
    instances, parity, parity_feasible = {}, {}, {}
    for name in CHENNAI:
        folder = ROOT / "data" / name
        inst = load_instance(folder, name)
        instances[name] = instance_block(inst, folder)
        parity[name] = parity_cases(inst)
        parity_feasible[name] = feasible_cases(inst, name)

    doc = {
        "_comment": (
            "Solver core for a browser-side solver of the Gonzalez et al. "
            "(2025) model, exported from the Python reference "
            "(src/export_solver_core.py). Floats are unrounded: Python's "
            "shortest repr, which JSON.parse reads back to the identical "
            "double. Rupee figures are an FX conversion of the paper's US$ "
            "cost parameters. Check it against the Python code with "
            "scripts/verify_js_fixtures.py."),
        "format_version": 1,
        "decoder_spec": DECODER_SPEC,
        "instances": instances,
        "validation": validation_block(),
        "parity": {
            "_comment": (f"{PARITY_CASES} random (pop, mask) pairs per "
                         "instance (optimizers.random_solution, repaired "
                         "masks) with the exact Python decode, penalties at "
                         "the instance's lam/gamma. Stops are 0-based "
                         "points."),
            **parity},
        "parity_feasible": {
            "_comment": ("Random schedules on these instances are all "
                         "infeasible (they break TL or the fleet size), so "
                         "these are the saved best solutions of the 30 SA "
                         "and 30 GA runs per instance (all feasible, penalty "
                         "0), with the same exact Python decode."),
            **parity_feasible},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
                   + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
