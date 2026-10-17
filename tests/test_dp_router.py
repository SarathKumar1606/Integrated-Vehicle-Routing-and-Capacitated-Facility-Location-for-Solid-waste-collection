"""
Correctness tests for src/dp_router.py.

  1. Exactness: on days of 4-8 points (synthetic and taken from the real
     instances), enumerate every partition into routes and every visiting
     order, and check that DP finds the true optimum EXACTLY (==, not a
     tolerance), and that it reports infeasibility exactly when no partition
     is feasible.
  2. polish() on >= 200 random solutions per instance: overall cost and
     fitness never increase, bin_cost is byte-identical, every replaced day
     respects capacity, TL and fleet size, and a feasible input stays feasible.
  3. Independent check of the polished cost: evaluate_explicit_routes()
     re-scores the explicit routes with the decoder's timing and cost code and
     must reproduce the polished routing cost exactly, and agree with the DP's
     own reported day totals (to 1e-9 min: DP adds service times in a
     different order, see below).

Polished solutions are stored as explicit routes, not as a chromosome: a DP
partition is not always a greedy split of some visiting order (the decoder
fills a truck in permutation order and cuts when the next point would
overflow it), so it cannot in general be written back into (pop, mask).

Why exact equality is possible in test 1: the brute force adds numbers in
the same order as the DP -- travel legs in visiting order, a route's service
times and loads in descending point order, route totals folded from the right
in lowest-point order -- and floating-point rounding is monotonic, so the
minimum over all orders and partitions equals the DP's value bit for bit.

Run:  python tests/test_dp_router.py [random_solutions_per_instance]
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from decoder import decode                                   # noqa: E402
from dp_router import (MAX_DP_POINTS, dp_day_routes,         # noqa: E402
                       evaluate_explicit_routes, held_karp_all_subsets,
                       polish_with_report, solve_day)
from loader import load_instance                             # noqa: E402
from optimizers import penalty_weights, random_solution      # noqa: E402

INSTANCES = ["12_1", "chennai_guindy", "chennai_guindy_peak"]
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)
        if len(FAILURES) <= 10:
            print(f"  FAIL: {msg}")


# ---------------------------------------------------------------------------
# Brute force
# ---------------------------------------------------------------------------
def set_partitions(items: list[int]):
    """All partitions of items into blocks, each block listing its items in
    ascending order and blocks ordered by their lowest item."""
    for part in _partitions(items):
        yield sorted((sorted(b) for b in part), key=lambda b: b[0])


def _partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _partitions(rest):
        yield [[first]] + part
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]


def brute_closed(block, pts, C, svc, tu):
    """Best closed tour over all visiting orders, summed like the DP."""
    best = np.inf
    for perm in itertools.permutations(block):
        t = C[0, pts[perm[0]] + 1]
        for a, b in zip(perm, perm[1:]):
            t += C[pts[a] + 1, pts[b] + 1]
        t += C[pts[perm[-1]] + 1, 0]
        best = min(best, t)
    s = 0.0
    for j in sorted(block, reverse=True):
        s += svc[j]
    return best + s + tu


def brute_day(pts, waste, svc, C, Q, TL, K, tu):
    n = len(pts)
    closed = {}
    for r in range(1, n + 1):
        for block in itertools.combinations(range(n), r):
            closed[block] = brute_closed(block, pts, C, svc, tu)

    def load(block):
        x = 0.0
        for j in sorted(block, reverse=True):
            x += waste[j]
        return x

    best = np.inf
    for part in set_partitions(list(range(n))):
        if len(part) > K:
            continue
        ok = all(load(b) <= Q + 1e-9 and closed[tuple(b)] <= TL + 1e-9
                 for b in part)
        if not ok:
            continue
        total = 0.0
        for b in reversed(part):              # f(S,k) = closed(T) + f(S-T,k-1)
            total = closed[tuple(b)] + total
        best = min(best, total)
    return best, closed


def check_day(pts, waste, bins, C, service, Q, TL, K, tu, label):
    pts = np.asarray(pts, dtype=np.int64)
    waste = np.asarray(waste, dtype=np.float64)
    svc = np.array([service[bins[p]] for p in pts])
    want, closed = brute_day(pts, waste, svc, C, Q, TL, K, tu)

    hk_closed = held_karp_all_subsets(pts, C, svc, tu)[0]
    for block, val in closed.items():
        mask = sum(1 << j for j in block)
        check(hk_closed[mask] == val,
              f"{label}: Held-Karp {hk_closed[mask]!r} != brute {val!r} for {block}")

    ok, got, nr, ptr, stops = dp_day_routes(pts, waste, bins, C, service,
                                            Q, TL, K, tu)
    if want == np.inf:
        check(not ok, f"{label}: DP found a partition but none is feasible")
        return
    check(ok, f"{label}: DP infeasible but brute force found {want}")
    check(got == want, f"{label}: DP {got!r} != brute force optimum {want!r}")
    # the returned routes are a partition of the day's points into <= K routes
    served = sorted(stops[:ptr[nr]].tolist())
    check(served == sorted(pts.tolist()), f"{label}: routes do not cover the day")
    check(nr <= K, f"{label}: {nr} routes > fleet {K}")


# ---------------------------------------------------------------------------
# Test 1: DP == brute force on 4-8 point days
# ---------------------------------------------------------------------------
def test_exact_synthetic(rng, cases_per_size=12):
    n_cases = 0
    for n in range(4, 9):
        for _ in range(cases_per_size):
            N = n + 3                                     # a few unused points
            C = np.round(rng.uniform(2, 15, size=(N + 1, N + 1)), 2)
            np.fill_diagonal(C, 0.0)
            if rng.random() < 0.5:                        # symmetric half the time
                C = np.round((C + C.T) / 2, 2)
            pts = np.sort(rng.choice(N, size=n, replace=False))
            waste = np.round(rng.uniform(0.3, 2.5, size=n), 2)
            bins = rng.integers(0, 3, size=N)
            service = np.array([0.7, 1.4, 1.33])
            Q = float(rng.choice([3.0, 4.5, 6.0, 12.0]))
            TL = float(rng.choice([25.0, 40.0, 60.0, 500.0]))
            K = int(rng.integers(1, 4))
            check_day(pts, waste, bins, C, service, Q, TL, K, 8.0,
                      f"synthetic n={n}")
            n_cases += 1
    return n_cases


def test_exact_real_days(rng, per_instance=15):
    """Days of 4-8 points from decoded random solutions on the real
    instances, with their real matrices, loads, bins, Q, TL and fleet."""
    n_cases = 0
    for name in INSTANCES:
        inst = load_instance(ROOT / "data" / name)
        lam, gamma = penalty_weights(inst)
        found = 0
        while found < per_instance:
            pop, mask = random_solution(inst, rng)
            sol = decode(inst, pop, mask, lam, gamma)
            for day in inst.work_days:
                pts = sorted(s for r in sol.routes_on(day) for s in r.stops)
                if 4 <= len(pts) <= 8 and found < per_instance:
                    check_day(pts, [sol.w[p, day] for p in pts], sol.bins, inst.C,
                              inst.bin_service, float(inst.Q), float(inst.TL),
                              int(inst.n_vehicles), float(inst.tu),
                              f"{name} day {day} ({len(pts)} pts)")
                    found += 1
                    n_cases += 1
    return n_cases


# ---------------------------------------------------------------------------
# Tests 2 + 3: polish on random solutions
# ---------------------------------------------------------------------------
def test_polish(rng, n_solutions):
    rows = []
    for name in INSTANCES:
        inst = load_instance(ROOT / "data" / name)
        lam, gamma = penalty_weights(inst)
        improved = skipped = 0
        for k in range(n_solutions):
            pop, mask = random_solution(inst, rng)
            sol = decode(inst, pop, mask, lam, gamma)
            pol, rep = polish_with_report(inst, sol, lam, gamma)
            tag = f"{name} solution {k}"

            check(pol.overall_cost <= sol.overall_cost,
                  f"{tag}: overall cost rose {sol.overall_cost} -> {pol.overall_cost}")
            check(pol.routing_cost <= sol.routing_cost, f"{tag}: routing cost rose")
            check(pol.fitness <= sol.fitness, f"{tag}: fitness rose")
            check(float(pol.bin_cost).hex() == float(sol.bin_cost).hex()
                  and pol.bins.tobytes() == sol.bins.tobytes()
                  and pol.w.tobytes() == sol.w.tobytes(),
                  f"{tag}: bin cost, bins or waste changed")
            if sol.feasible:
                check(pol.feasible, f"{tag}: feasible input became infeasible")

            for d in rep.days:
                if d.status != "improved":
                    continue
                routes = pol.routes_on(d.day)
                check(len(routes) <= inst.n_vehicles, f"{tag}: fleet exceeded")
                for r in routes:
                    check(r.total_load <= inst.Q + 1e-9, f"{tag}: capacity exceeded")
                    check(r.duration <= inst.TL + 1e-9, f"{tag}: TL exceeded")
                # the DP's own reported total for this day
                pts = sorted(s for r in sol.routes_on(d.day) for s in r.stops)
                _, dp_total, _, _ = solve_day(inst, sol.bins, pts,
                                              [sol.w[p, d.day] for p in pts])
                check(abs(dp_total - d.new_minutes) <= 1e-9,
                      f"{tag}: DP total {dp_total} vs decoder timing {d.new_minutes}")

            # independent re-scoring of the explicit routes
            score = evaluate_explicit_routes(
                inst, pol.bins, pol.w, [(r.day, r.stops) for r in pol.routes],
                lam, gamma)
            check(score.routing_cost == pol.routing_cost
                  and score.penalty == pol.penalty
                  and score.bin_cost == pol.bin_cost,
                  f"{tag}: evaluate_explicit_routes disagrees with polish")
            check(score.durations == [r.duration for r in pol.routes],
                  f"{tag}: route durations disagree")
            # and on the unpolished solution it must reproduce decode() exactly
            raw = evaluate_explicit_routes(
                inst, sol.bins, sol.w, [(r.day, r.stops) for r in sol.routes],
                lam, gamma)
            check(raw.routing_cost == sol.routing_cost and raw.penalty == sol.penalty,
                  f"{tag}: evaluate_explicit_routes disagrees with decode()")

            improved += rep.changed
            skipped += len(rep.skipped)
        rows.append((name, n_solutions, improved, skipped))
    return rows


def main() -> int:
    n_sol = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = np.random.default_rng(1729)
    t0 = time.perf_counter()
    print(f"MAX_DP_POINTS = {MAX_DP_POINTS}")
    n1 = test_exact_synthetic(rng)
    print(f"  exact vs brute force, synthetic days (4-8 points): {n1} cases")
    n2 = test_exact_real_days(rng)
    print(f"  exact vs brute force, real-instance days (4-8 points): {n2} cases")
    for name, n, imp, skip in test_polish(rng, n_sol):
        print(f"  polish {name:<20} {n} random solutions, {imp} improved, "
              f"{skip} days skipped (> {MAX_DP_POINTS} points)")
    print(f"  ({time.perf_counter() - t0:.0f} s)")
    print(f"RESULT: {'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILURES'}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
