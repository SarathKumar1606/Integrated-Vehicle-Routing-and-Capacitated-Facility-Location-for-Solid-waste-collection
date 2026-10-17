"""
DP router: exact routing for a single day, holding the day assignment fixed
=========================================================================
An opt-in layer on top of SA/GA. The metaheuristics decide *which* points
are collected on *which* day (the mask) and therefore the bins and the
waste loads. The decoder then cuts each day's visiting order greedily into
capacity-feasible routes. This module replaces that greedy split, day by
day, with the exact optimum for the same points:

  1. held_karp_all_subsets()  for every subset T of the day's points, the
                              minimum closed-tour time depot -> T -> depot,
                              including service times and the unload time.
  2. dp_day_routes()          the minimum-total-time partition of the day's
                              points into at most nV admissible routes
                              (load <= Q, closed tour <= TL).
  3. polish()                 applies 2. to every working day of a decoded
                              solution. Bins, waste accumulation and the mask
                              are unchanged, so bin_cost is identical and only
                              routing_cost can move -- never upwards.

Nothing here is imported by the decoder, the optimizers or the Appendix A
validation: SA and GA results are unaffected unless --polish is passed.

Complexity (n = points collected that day, K = fleet size)
  Held-Karp       O(n^2 2^n) time, O(n 2^n) memory
  partition DP    O(K 3^n) time (each set S enumerates the subsets of S that
                  contain its lowest element), O(K 2^n) memory
MAX_DP_POINTS = 14 keeps both well under a second; larger days keep the
decoder's greedy routes and are reported as skipped.

Floating point: DP sums travel legs in visiting order, like the decoder, but
adds a route's service times in a different order, so its internal total can
differ from the decoder's in the last bits. polish() therefore re-times every
route with the decoder's own _route_duration, and only accepts a day if it
improves by more than ACCEPT_TOL minutes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from decoder import Route, Solution, _route_duration
from loader import DAY_NAMES, Instance
from optimizers import penalty_weights

MAX_DP_POINTS = 14
ACCEPT_TOL = 1e-9           # minutes; smaller "improvements" are rounding noise


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@njit(cache=True)
def _low_index(S):
    """Index of the lowest set bit of S (S > 0)."""
    j = 0
    while not (S >> j) & 1:
        j += 1
    return j


@njit(cache=True)
def held_karp_all_subsets(pts, C, svc, tu):
    """Held-Karp over every subset of the day's points.

    pts[j] is a collection-point index (row/col pts[j] + 1 of C; 0 is the
    depot) and svc[j] its bin's service time. dp[S, j] is the minimum travel
    time leaving the depot, visiting exactly the points in bitmask S and
    ending at j:

        dp[{j}, j]      = C[0, j]
        dp[S + {k}, k]  = min over j in S of  dp[S, j] + C[j, k]

    closed[S] = min_j (dp[S, j] + C[j, 0]) + service(S) + tu is the duration of
    the best single route serving S, as Eq. (1) defines it. end[S] and
    parent[S, j] let the visiting order be read back.
    """
    n = pts.shape[0]
    full = 1 << n
    dp = np.full((full, n), np.inf)
    parent = np.full((full, n), -1, dtype=np.int8)
    for j in range(n):
        dp[1 << j, j] = C[0, pts[j] + 1]
    for S in range(1, full):
        for j in range(n):
            if not (S >> j) & 1:
                continue
            base = dp[S, j]
            if base == np.inf:
                continue
            for k in range(n):
                if (S >> k) & 1:
                    continue
                T = S | (1 << k)
                cand = base + C[pts[j] + 1, pts[k] + 1]
                if cand < dp[T, k]:
                    dp[T, k] = cand
                    parent[T, k] = j

    closed = np.full(full, np.inf)
    end = np.full(full, -1, dtype=np.int8)
    service = np.zeros(full)
    closed[0] = 0.0
    for S in range(1, full):
        lj = _low_index(S)
        service[S] = service[S ^ (1 << lj)] + svc[lj]
        best, bj = np.inf, -1
        for j in range(n):
            if (S >> j) & 1:
                c = dp[S, j] + C[pts[j] + 1, 0]
                if c < best:
                    best, bj = c, j
        closed[S] = best + service[S] + tu
        end[S] = bj
    return closed, dp, parent, end


@njit(cache=True)
def _tour_order(S, end, parent, out):
    """Write the optimal visiting order of subset S (as local indices j)
    into out[0:popcount(S)] and return its length."""
    m = 0
    T = S
    while T:
        T &= T - 1
        m += 1
    j = end[S]
    T = S
    pos = m - 1
    while T:
        out[pos] = j
        pj = parent[T, j]
        T ^= 1 << j
        j = pj
        pos -= 1
    return m


@njit(cache=True)
def dp_day_routes(pts, waste, bins, C, service, Q, TL, n_vehicles, tu):
    """Exact minimum-total-time routes for one day's points.

    f(S, k) = minimum total time to serve the points in S with exactly k
    routes. A route T is admissible if load(T) <= Q and closed(T) <= TL. To
    count each partition once, the route taking the lowest-indexed point of
    S is chosen first:

        f(0, 0) = 0
        f(S, k) = min over admissible T in S with low(S) in T  of
                  closed(T) + f(S - T, k - 1)

    The answer is min over k <= n_vehicles of f(all, k).

    waste[j] is the load at pts[j] that day; service is indexed by bin
    (service[bins[pts[j]]]). Returns (feasible, total_minutes, n_routes,
    route_ptr, route_stops): route r visits the collection points
    route_stops[route_ptr[r]:route_ptr[r + 1]] in order.
    """
    n = pts.shape[0]
    route_ptr = np.zeros(n + 1, dtype=np.int64)
    route_stops = np.zeros(n, dtype=np.int64)
    if n == 0:
        return True, 0.0, 0, route_ptr, route_stops

    svc = np.empty(n)
    for j in range(n):
        svc[j] = service[bins[pts[j]]]
    closed, dp, parent, end = held_karp_all_subsets(pts, C, svc, tu)

    full = 1 << n
    load = np.zeros(full)
    adm = np.zeros(full, dtype=np.bool_)
    for S in range(1, full):
        lj = _low_index(S)
        load[S] = load[S ^ (1 << lj)] + waste[lj]
        adm[S] = load[S] <= Q + 1e-9 and closed[S] <= TL + 1e-9

    K = n_vehicles
    f = np.full((full, K + 1), np.inf)
    choice = np.zeros((full, K + 1), dtype=np.int64)
    f[0, 0] = 0.0
    for S in range(1, full):
        low = S & -S
        rest = S ^ low
        sub = rest
        while True:
            T = sub | low
            if adm[T]:
                R = S ^ T
                c = closed[T]
                for k in range(1, K + 1):
                    prev = f[R, k - 1]
                    if prev < np.inf:
                        v = c + prev
                        if v < f[S, k]:
                            f[S, k] = v
                            choice[S, k] = T
            if sub == 0:
                break
            sub = (sub - 1) & rest

    best, bk = np.inf, -1
    for k in range(1, K + 1):
        if f[full - 1, k] < best:
            best, bk = f[full - 1, k], k
    if bk < 0:
        return False, np.inf, 0, route_ptr, route_stops

    order = np.empty(n, dtype=np.int64)
    S, k, r, pos = full - 1, bk, 0, 0
    while S:
        T = choice[S, k]
        m = _tour_order(T, end, parent, order)
        for q in range(m):
            route_stops[pos] = pts[order[q]]
            pos += 1
        r += 1
        route_ptr[r] = pos
        S ^= T
        k -= 1
    return True, best, r, route_ptr, route_stops


# ---------------------------------------------------------------------------
# Scoring explicit routes with the decoder's own timing and cost code
# ---------------------------------------------------------------------------
@dataclass
class ExplicitScore:
    durations: list[float]
    loads: list[float]             # total load of each route
    bin_cost: float
    routing_cost: float
    overall_cost: float
    penalty: float
    total_minutes: float


def evaluate_explicit_routes(inst: Instance, bins: np.ndarray, w: np.ndarray,
                             routes: list[tuple[int, list[int]]],
                             lam: float | None = None,
                             gamma: float | None = None) -> ExplicitScore:
    """Score routes given explicitly as (day, [stops]) pairs, in that order.

    Uses the decoder's _route_duration (Eq. 1), its bin-cost sum, its
    route-by-route minute total, and its Eq. (7) penalty rules, so a set of
    routes that the decoder could produce scores exactly as decode() would.

    Why explicit routes at all: the decoder only produces routes that are a
    greedy split of a visiting order (fill the truck, cut when the next point
    would overflow it). A DP partition need not be of that form -- e.g. it may
    send a lighter route out while a heavier pair shares the next truck, which
    no permutation's greedy split yields. So a polished solution cannot in
    general be written back into a (pop, mask) chromosome; it is stored and
    scored as explicit routes instead.
    """
    if lam is None or gamma is None:
        lam, gamma = penalty_weights(inst)
    scratch = np.empty(max(inst.n_points, 8))
    durations, loads = [], []
    for day, stops in routes:
        arr = np.asarray(stops, dtype=np.int64)
        durations.append(float(_route_duration(inst.C, arr, 0, arr.size, bins,
                                               inst.bin_service, inst.tu, scratch)))
        load = 0.0
        for s in stops:
            load += w[s, day]
        loads.append(load)
    total = 0.0
    for d in durations:
        total += d
    bin_cost = float(inst.bin_cost[bins].sum())
    routing = inst.ccv * total
    penalty = 0.0
    for day in range(inst.n_days):
        n_routes = sum(1 for d, _ in routes if d == day)
        if n_routes > inst.n_vehicles:
            penalty += lam * (n_routes - inst.n_vehicles) / inst.n_vehicles
    for d in durations:
        if d > inst.TL + 1e-9:
            penalty += gamma * (d - inst.TL)
    return ExplicitScore(durations, loads, bin_cost, routing, bin_cost + routing,
                         penalty, total)


# ---------------------------------------------------------------------------
# Polish: DP routes for every working day of a decoded solution
# ---------------------------------------------------------------------------
@dataclass
class DayReport:
    day: int
    n_points: int
    status: str            # improved | unchanged | skipped_size | no_feasible_dp
                           # | rejected_check
    greedy_minutes: float
    new_minutes: float
    dp_ms: float           # wall time of dp_day_routes for this day


@dataclass
class PolishReport:
    days: list[DayReport] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(d.status == "improved" for d in self.days)

    @property
    def skipped(self) -> list[DayReport]:
        return [d for d in self.days if d.status == "skipped_size"]


def solve_day(inst: Instance, bins: np.ndarray, pts: list[int],
              loads: list[float]) -> tuple[bool, float, list[list[int]], float]:
    """Run dp_day_routes for one day. Returns (feasible, dp_total_minutes,
    routes as lists of collection points, wall time in ms)."""
    p = np.asarray(pts, dtype=np.int64)
    wl = np.asarray(loads, dtype=np.float64)
    t0 = time.perf_counter()
    ok, total, nr, ptr, stops = dp_day_routes(
        p, wl, bins, inst.C, inst.bin_service, float(inst.Q), float(inst.TL),
        int(inst.n_vehicles), float(inst.tu))
    ms = (time.perf_counter() - t0) * 1000.0
    routes = [stops[ptr[r]:ptr[r + 1]].tolist() for r in range(nr)]
    return bool(ok), float(total), routes, ms


def _make_route(day: int, stops: list[int], w: np.ndarray,
                duration: float) -> Route:
    loads, cum, load = [], [], 0.0
    for s in stops:
        amount = float(w[s, day])
        load += amount
        loads.append(amount)
        cum.append(load)
    return Route(day=day, stops=list(stops), loads=loads, cum_load=cum,
                 duration=duration)


def _assemble(inst: Instance, sol: Solution, routes: list[Route],
              lam: float, gamma: float) -> Solution:
    """A Solution with the input's bins and waste and the given routes."""
    score = evaluate_explicit_routes(inst, sol.bins, sol.w,
                                     [(r.day, r.stops) for r in routes], lam, gamma)
    violations = []
    for day in range(inst.n_days):
        n_routes = sum(1 for r in routes if r.day == day)
        if n_routes > inst.n_vehicles:
            violations.append(
                f"{DAY_NAMES[day]}: {n_routes} routes > fleet of {inst.n_vehicles}")
    for r in routes:
        if r.duration > inst.TL + 1e-9:
            violations.append(
                f"{DAY_NAMES[r.day]}: route {r.duration:.2f} min > TL {inst.TL:.2f}")
    return Solution(
        instance_name=sol.instance_name, bins=sol.bins, w=sol.w, wmax=sol.wmax,
        routes=routes, bin_cost=sol.bin_cost,      # unchanged, byte for byte
        routing_cost=score.routing_cost,
        overall_cost=sol.bin_cost + score.routing_cost,
        feasible=(score.penalty == 0.0), penalty=score.penalty,
        violations=violations, currency=sol.currency,
        currency_symbol=sol.currency_symbol)


def polish_with_report(inst: Instance, sol: Solution,
                       lam: float | None = None,
                       gamma: float | None = None) -> tuple[Solution, PolishReport]:
    """polish(), plus what happened on each working day."""
    if lam is None or gamma is None:
        lam, gamma = penalty_weights(inst)
    report = PolishReport()
    scratch = np.empty(max(inst.n_points, 8))
    new_routes: list[Route] = []
    for day in inst.work_days:
        greedy = sol.routes_on(day)
        pts = [s for r in greedy for s in r.stops]
        g_min = 0.0
        for r in greedy:
            g_min += r.duration
        if not pts:
            continue
        if len(pts) > MAX_DP_POINTS:
            report.days.append(DayReport(day, len(pts), "skipped_size",
                                         g_min, g_min, 0.0))
            new_routes += greedy
            continue

        ok, _, dp_routes, ms = solve_day(inst, sol.bins, pts,
                                         [float(sol.w[s, day]) for s in pts])
        if not ok:
            report.days.append(DayReport(day, len(pts), "no_feasible_dp",
                                         g_min, g_min, ms))
            new_routes += greedy
            continue

        # re-time with the decoder's own code and check every constraint
        cand = []
        for stops in dp_routes:
            arr = np.asarray(stops, dtype=np.int64)
            dur = float(_route_duration(inst.C, arr, 0, arr.size, sol.bins,
                                        inst.bin_service, inst.tu, scratch))
            cand.append(_make_route(day, stops, sol.w, dur))
        n_min = 0.0
        for r in cand:
            n_min += r.duration
        valid = (len(cand) <= inst.n_vehicles
                 and all(r.total_load <= inst.Q + 1e-9 for r in cand)
                 and all(r.duration <= inst.TL + 1e-9 for r in cand))
        if valid and n_min < g_min - ACCEPT_TOL:
            report.days.append(DayReport(day, len(pts), "improved",
                                         g_min, n_min, ms))
            new_routes += cand
        else:
            status = "unchanged" if valid else "rejected_check"
            report.days.append(DayReport(day, len(pts), status, g_min, g_min, ms))
            new_routes += greedy

    polished = _assemble(inst, sol, new_routes, lam, gamma)
    if polished.routing_cost > sol.routing_cost:   # cannot happen; belt and braces
        return sol, PolishReport([DayReport(d.day, d.n_points, "unchanged",
                                            d.greedy_minutes, d.greedy_minutes,
                                            d.dp_ms) for d in report.days])
    return polished, report


def polish(inst: Instance, solution: Solution) -> Solution:
    """Replace each working day's greedy routes by the exact DP optimum for
    the same points. Days with more than MAX_DP_POINTS points keep their
    greedy routes (see polish_with_report for which). The result has the
    same bins, waste and bin_cost, and routing_cost <= the input's."""
    return polish_with_report(inst, solution)[0]


# ---------------------------------------------------------------------------
# Dynamic re-planning of one day
# ---------------------------------------------------------------------------
@dataclass
class ReplanResult:
    day: int
    routes: list[Route]
    old_minutes: float
    new_minutes: float
    cost_change: float             # in the instance's currency (CCV x minutes)
    method: str                    # "dp" or "greedy" (day too large for DP)
    feasible: bool
    ms: float


def replan_day(inst: Instance, solution: Solution, day: int,
               revised_waste: dict[int, float]) -> ReplanResult:
    """Re-solve one day after new waste figures arrive.

    revised_waste maps a collection point (0-based) to the waste, in m3, it
    will hold when collected on `day`. A point that was not scheduled that day
    is added to it (e.g. a bin filling early after a surge); a scheduled point
    gets its new load. Every other point keeps the plan's figure. The day is
    re-routed exactly with DP; bins and the other days are left as planned.

    Limitations: only this day is re-planned. Collecting a point earlier also
    lowers its load on its next scheduled day; that is not propagated, so the
    rest of the week is left on the conservative side. A revised load larger
    than the vehicle capacity Q cannot be served by any single route.
    """
    old = solution.routes_on(day)
    pts = [s for r in old for s in r.stops]
    for p in revised_waste:
        if p not in pts:
            pts.append(int(p))
    pts = sorted(pts)
    loads = [float(revised_waste.get(s, solution.w[s, day])) for s in pts]
    old_min = 0.0
    for r in old:
        old_min += r.duration

    t0 = time.perf_counter()
    if len(pts) <= MAX_DP_POINTS:
        ok, _, dp_routes, _ = solve_day(inst, solution.bins, pts, loads)
        method = "dp"
    else:                          # fall back to a greedy split of the points
        ok, dp_routes, cur, load = True, [], [], 0.0
        for s, a in zip(pts, loads):
            if cur and load + a > inst.Q + 1e-9:
                dp_routes.append(cur)
                cur, load = [], 0.0
            cur.append(s)
            load += a
        if cur:
            dp_routes.append(cur)
        method = "greedy"
    ms = (time.perf_counter() - t0) * 1000.0

    scratch = np.empty(max(inst.n_points, 8))
    load_of = dict(zip(pts, loads))
    routes = []
    for stops in dp_routes:
        arr = np.asarray(stops, dtype=np.int64)
        dur = float(_route_duration(inst.C, arr, 0, arr.size, solution.bins,
                                    inst.bin_service, inst.tu, scratch))
        cum, ld, la = [], 0.0, []
        for s in stops:
            ld += load_of[s]
            la.append(load_of[s])
            cum.append(ld)
        routes.append(Route(day=day, stops=list(stops), loads=la, cum_load=cum,
                            duration=dur))
    new_min = 0.0
    for r in routes:
        new_min += r.duration
    feasible = ok and len(routes) <= inst.n_vehicles and all(
        r.total_load <= inst.Q + 1e-9 and r.duration <= inst.TL + 1e-9
        for r in routes)
    return ReplanResult(day, routes, old_min, new_min,
                        inst.ccv * (new_min - old_min), method, feasible, ms)


# ---------------------------------------------------------------------------
# Recording polished solutions (runner.py / exporter.py / polish_experiment.py)
# ---------------------------------------------------------------------------
def routes_to_arrays(routes: list[Route]) -> dict[str, np.ndarray]:
    """Explicit routes as flat arrays for np.savez: route r runs on
    route_day[r] and visits route_stops[route_ptr[r]:route_ptr[r + 1]]."""
    ptr = np.zeros(len(routes) + 1, dtype=np.int64)
    for r, route in enumerate(routes):
        ptr[r + 1] = ptr[r] + len(route.stops)
    stops = np.array([s for r in routes for s in r.stops], dtype=np.int64)
    days = np.array([r.day for r in routes], dtype=np.int64)
    return {"polished_route_day": days, "polished_route_ptr": ptr,
            "polished_route_stops": stops}


_WARM = False


def warm_up() -> None:
    """Compile (or load from the on-disk cache) the DP kernels once, so the
    first timed day does not include several seconds of JIT start-up."""
    global _WARM
    if _WARM:
        return
    C = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    dp_day_routes(np.array([0, 1], dtype=np.int64), np.array([1.0, 1.0]),
                  np.array([0, 0], dtype=np.int64), C, np.array([1.0]),
                  10.0, 100.0, 2, 1.0)
    _WARM = True


def polish_record(inst: Instance, sol: Solution) -> tuple[Solution, dict, dict]:
    """Polish a decoded solution and return (polished solution, extra CSV
    columns, extra arrays to save alongside the chromosome)."""
    warm_up()
    t0 = time.perf_counter()
    pol, rep = polish_with_report(inst, sol)
    ms = (time.perf_counter() - t0) * 1000.0
    dp_days = [d for d in rep.days if d.status != "skipped_size"]
    columns = {
        "polished_overall_cost": round(pol.overall_cost, 4),
        "polished_routing_cost": round(pol.routing_cost, 4),
        "polished_feasible": pol.feasible,
        "polish_days_improved": sum(d.status == "improved" for d in rep.days),
        "polish_days_skipped": len(rep.skipped),
        "polish_dp_ms_per_day": round(sum(d.dp_ms for d in dp_days)
                                      / max(len(dp_days), 1), 3),
        "polish_ms": round(ms, 2),
    }
    return pol, columns, routes_to_arrays(pol.routes)
