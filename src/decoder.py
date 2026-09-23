"""
M4 - Decoder and Evaluator  (the heart of the project)
======================================================
Takes a candidate solution and works out what it would actually cost.

A candidate solution is two matrices, both shape (nI, n_work_days):

  pop[i, d]  : the position of collection point i in day d's visiting order.
               Each column is a permutation of 0 .. nI-1.
  mask[i, d] : 1 if point i is collected on day d, 0 otherwise.

The decoder runs five stages:

  1. _accumulate()    simulate the week, cyclically, to get w[i, t]
  2. _choose_bins()   pick the cheapest workable bin combination per point
  3. _decode_core()   cut each day's visiting order into capacity-feasible routes
  4. _route_duration() travel time + service time + unloading time
  5. cost + penalties

Stage 1's cyclic wrap-around (Eq. 2l) is the subtle part: Monday's starting
pile depends on what was left over from Sunday, and the schedule repeats every
week, so we iterate the week until the accumulation stops changing.

Performance
-----------
The optimizers call the decoder millions of times, so stages 1-5 are compiled
with Numba (`@njit`) and work on flat arrays. `evaluate()` returns only the
fitness and is what SA and GA call; `decode()` additionally unpacks the result
into `Route` objects and violation messages for reporting.

The compiled code is written to give *bit-identical* results to the original
pure-Python/NumPy decoder (kept as tests/reference_decoder.py and checked by
tests/test_decoder_equivalence.py). That is why `_pairwise_sum` reproduces
NumPy's own summation order and `_accumulate` reproduces `np.allclose`'s
default tolerances: a JIT-compiled loop that summed in a different order
would drift in the last bits and could flip an optimizer's decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from loader import DAY_NAMES, Instance


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class Route:
    day: int
    stops: list[int]               # collection point indices, in visiting order
    loads: list[float]             # waste picked up at each stop
    cum_load: list[float]          # cumulative load after each stop
    duration: float                # minutes, depot to depot, incl. unloading

    @property
    def total_load(self) -> float:
        return self.cum_load[-1] if self.cum_load else 0.0


@dataclass
class Solution:
    instance_name: str
    bins: np.ndarray               # bins[i] = chosen bin combination for point i
    w: np.ndarray                  # w[i, t] accumulated waste, shape (nI, 7)
    wmax: np.ndarray               # wmax[i] = worst day of the week
    routes: list[Route]
    bin_cost: float
    routing_cost: float
    overall_cost: float
    feasible: bool
    penalty: float = 0.0
    violations: list[str] = field(default_factory=list)

    @property
    def fitness(self) -> float:
        """What the optimizer minimizes: true cost plus penalties."""
        return self.overall_cost + self.penalty

    def routes_on(self, day: int) -> list[Route]:
        return [r for r in self.routes if r.day == day]

    def report(self) -> str:
        lines = [f"Solution for {self.instance_name}",
                 f"  bin cost     : {self.bin_cost:8.2f} US$",
                 f"  routing cost : {self.routing_cost:8.2f} US$",
                 f"  OVERALL cost : {self.overall_cost:8.2f} US$",
                 f"  feasible     : {self.feasible}"]
        if self.violations:
            lines += [f"    ! {v}" for v in self.violations]
        lines.append("  routes:")
        for day in range(7):
            for k, r in enumerate(self.routes_on(day), start=1):
                seq = " -> ".join(["0"] + [str(s + 1) for s in r.stops] + ["0"])
                lines.append(f"    {DAY_NAMES[day]} R{k}: {seq}"
                             f"   load {r.total_load:5.2f}   {r.duration:6.2f} min")
        return "\n".join(lines)


NO_BIN_FITS = ("No bin combination is large enough for some point. "
               "The instance needs a bigger bin or more frequent visits.")


# ---------------------------------------------------------------------------
# Summation in NumPy's order
# ---------------------------------------------------------------------------
@njit(cache=True)
def _pairwise_block(a, lo, n):
    """NumPy's summation of a block of at most 128 elements."""
    if n < 8:
        res = -0.0
        for i in range(n):
            res += a[lo + i]
        return res
    if n <= 128:
        r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
        r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
        i = 8
        while i < n - (n % 8):
            r0 += a[lo + i]
            r1 += a[lo + i + 1]
            r2 += a[lo + i + 2]
            r3 += a[lo + i + 3]
            r4 += a[lo + i + 4]
            r5 += a[lo + i + 5]
            r6 += a[lo + i + 6]
            r7 += a[lo + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[lo + i]
            i += 1
        return res
    return np.nan                                    # n > 128: not a block


@njit(cache=True)
def _pairwise_sum(a, lo, n):
    """Sum a[lo:lo+n] exactly as `ndarray.sum()` does for float64: a plain
    loop below 8 elements, eight interleaved accumulators up to 128, and
    recursive halving beyond that (n2 = n//2 rounded down to a multiple of 8,
    result = sum(left) + sum(right)). Floating-point addition is not
    associative, so matching the order is what keeps results bit-identical to
    NumPy. The halving is done with an explicit stack because Numba cannot
    cache self-recursive functions."""
    # NumPy seeds the reduction with the identity +0.0 (so an all -0.0
    # input sums to +0.0, not -0.0).
    if n <= 128:
        return 0.0 + _pairwise_block(a, lo, n)
    f_lo = np.empty(64, dtype=np.int64)              # frames: (lo, n, state)
    f_n = np.empty(64, dtype=np.int64)
    f_state = np.empty(64, dtype=np.int64)
    vals = np.empty(64)                              # finished partial sums
    f_lo[0], f_n[0], f_state[0] = lo, n, 0
    sp, vp = 1, 0
    while sp > 0:
        top = sp - 1
        m = f_n[top]
        if m <= 128:
            vals[vp] = _pairwise_block(a, f_lo[top], m)
            vp += 1
            sp -= 1
            continue
        n2 = m // 2
        n2 -= n2 % 8
        if f_state[top] == 0:                        # descend left
            f_state[top] = 1
            f_lo[sp], f_n[sp], f_state[sp] = f_lo[top], n2, 0
            sp += 1
        elif f_state[top] == 1:                      # descend right
            f_state[top] = 2
            f_lo[sp], f_n[sp], f_state[sp] = f_lo[top] + n2, m - n2, 0
            sp += 1
        else:                                        # left + right
            vals[vp - 2] = vals[vp - 2] + vals[vp - 1]
            vp -= 1
            sp -= 1
    return 0.0 + vals[0]


# ---------------------------------------------------------------------------
# Stage 1 - cyclic waste accumulation  (Eqs. 2k, 2l)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _accumulate(W, mask, work_days, n_days, max_iter):
    """Return w[i, t] = waste sitting at point i at the END of day t.

    Waste is generated every day (including rest days) and is collected at the
    end of the day, so a visit on day t removes exactly w[i, t]. Because the
    weekly schedule repeats forever, we sweep the week repeatedly until the
    numbers settle -- that enforces the cyclic constraint Eq. (2l).

    "Settled" is np.allclose(carry, prev, atol=1e-12) with its default
    rtol=1e-5, i.e. |carry - prev| <= 1e-12 + 1e-5 * |prev| for every point.
    """
    nI = W.shape[0]

    # visited[i, t] over the FULL week; rest days are never collection days
    visited = np.zeros((nI, n_days), dtype=np.bool_)
    for col in range(work_days.shape[0]):
        for i in range(nI):
            visited[i, work_days[col]] = mask[i, col] == 1

    w = np.zeros((nI, n_days))
    carry = W.copy()                                 # arbitrary warm start
    prev = np.empty(nI)
    for _ in range(max_iter):
        prev[:] = carry
        for t in range(n_days):
            for i in range(nI):
                w[i, t] = W[i] + carry[i]            # today's generation + pile
                carry[i] = 0.0 if visited[i, t] else w[i, t]
        settled = True
        for i in range(nI):
            x, y = carry[i], prev[i]
            if not (x == y or (abs(x - y) <= 1e-12 + 1e-5 * abs(y)
                               and np.isfinite(y))):
                settled = False
                break
        if settled:
            break
    return w


# ---------------------------------------------------------------------------
# Stage 2 - bin selection  (Eqs. 2b, 2c)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _choose_bins(wmax, n_visits, bin_cap, bin_service, bin_cost, ccv):
    """Pick, for each point, the bin combination that minimizes total weekly
    cost among those big enough to hold wmax[i]. Returns (bins, ok); ok is
    False if some point fits in no bin at all.

    Important: it is NOT simply the cheapest bin that fits. A cheaper bin can
    have a longer service time, and that service time is paid on every visit
    at CCV US$/min. So the real objective per point is

        CIN_b  +  CCV * S_b * (number of visits that week)

    This is exactly what the paper's objective function (2) implies once
    service time is bin-dependent, and it is what reproduces its Appendix A
    bin choices (e.g. point 9 gets bin 4, not the cheaper-but-slower bin 3).
    Ties go to the lowest bin index, as with np.argmin.
    """
    nI, nB = wmax.shape[0], bin_cap.shape[0]
    bins = np.zeros(nI, dtype=np.int64)
    ok = True
    for i in range(nI):
        best, best_cost = 0, np.inf
        for b in range(nB):
            if bin_cap[b] >= wmax[i] - 1e-9:
                cost = bin_cost[b] + ccv * bin_service[b] * n_visits[i]
            else:
                cost = np.inf
            if b == 0 or cost < best_cost:
                best, best_cost = b, cost
        bins[i] = best
        if not np.isfinite(best_cost):
            ok = False
    return bins, ok


# ---------------------------------------------------------------------------
# Stage 4 - route timing  (Eq. 1)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _route_duration(C, stops, a, b, bins, bin_service, tu, scratch):
    """Duration of the route stops[a:b]:

    duration = sum of travel times along  0 -> s1 -> ... -> sk -> 0
             + sum of service times of the bins at each stop
             + unloading time at the depot
    Matrix indices are +1 because index 0 of C is the depot.
    """
    travel = C[0, stops[a] + 1]
    for k in range(a, b - 1):
        travel += C[stops[k] + 1, stops[k + 1] + 1]
    travel += C[stops[b - 1] + 1, 0]

    for k in range(a, b):
        scratch[k - a] = bin_service[bins[stops[k]]]
    service = _pairwise_sum(scratch, 0, b - a)
    return travel + service + tu


# ---------------------------------------------------------------------------
# The full decode, compiled
# ---------------------------------------------------------------------------
@njit(cache=True)
def _decode_core(W, C, bin_cap, bin_service, bin_cost, ccv, tu, Q, TL,
                 n_vehicles, work_days, n_days, lam, gamma, pop, mask):
    nI, nD = mask.shape

    # --- stage 1: how much waste is where, each day -----------------------
    w = _accumulate(W, mask, work_days, n_days, 50)
    wmax = np.empty(nI)
    for i in range(nI):
        wmax[i] = w[i].max()

    # --- stage 2: what bins to install ------------------------------------
    n_visits = np.empty(nI)
    for i in range(nI):
        v = 0
        for d in range(nD):
            v += mask[i, d]
        n_visits[i] = v
    bins, ok = _choose_bins(wmax, n_visits, bin_cap, bin_service, bin_cost, ccv)

    # Routes are stored flat: route r visits stops[ptr[r]:ptr[r + 1]].
    max_stops = nI * nD
    route_day = np.zeros(max_stops, dtype=np.int64)
    ptr = np.zeros(max_stops + 1, dtype=np.int64)
    stops = np.zeros(max_stops, dtype=np.int64)
    loads = np.zeros(max_stops)
    cum = np.zeros(max_stops)
    durations = np.zeros(max_stops)
    if not ok:
        return (w, wmax, bins, False, 0, route_day, ptr, stops, loads, cum,
                durations, 0.0, 0.0, 0.0, 0.0)

    # --- stages 3 + 4: build and time the routes --------------------------
    # Walk each day's visiting order, filling the truck. When the next point
    # would overflow the vehicle, close the route and start a new one.
    order = np.empty(nI, dtype=np.int64)
    scratch = np.empty(max(nI, 8))
    nr, ns = 0, 0
    for col in range(nD):
        day = work_days[col]

        # picked points, sorted by pop[., col] (stable insertion sort)
        cnt = 0
        for i in range(nI):
            if mask[i, col] == 1:
                j = cnt
                while j > 0 and pop[order[j - 1], col] > pop[i, col]:
                    order[j] = order[j - 1]
                    j -= 1
                order[j] = i
                cnt += 1
        if cnt == 0:
            continue

        start, load = ns, 0.0
        for k in range(cnt):
            i = order[k]
            amount = w[i, day]
            if ns > start and load + amount > Q + 1e-9:
                durations[nr] = _route_duration(C, stops, start, ns, bins,
                                                bin_service, tu, scratch)
                route_day[nr] = day
                ptr[nr + 1] = ns
                nr += 1
                start, load = ns, 0.0
            load += amount
            stops[ns] = i
            loads[ns] = amount
            cum[ns] = load
            ns += 1
        durations[nr] = _route_duration(C, stops, start, ns, bins,
                                        bin_service, tu, scratch)
        route_day[nr] = day
        ptr[nr + 1] = ns
        nr += 1

    # --- stage 5: cost and penalties --------------------------------------
    for i in range(nI):
        scratch[i] = bin_cost[bins[i]]
    total_bin_cost = _pairwise_sum(scratch, 0, nI)
    total_minutes = 0.0
    for r in range(nr):
        total_minutes += durations[r]
    routing_cost = ccv * total_minutes
    overall = total_bin_cost + routing_cost

    penalty = 0.0
    for day in range(n_days):                                # Eq. (2g)
        n_routes = 0
        for r in range(nr):
            if route_day[r] == day:
                n_routes += 1
        if n_routes > n_vehicles:
            penalty += lam * (n_routes - n_vehicles) / n_vehicles
    for r in range(nr):                                      # Eq. (2h)
        if durations[r] > TL + 1e-9:
            penalty += gamma * (durations[r] - TL)

    return (w, wmax, bins, True, nr, route_day, ptr, stops, loads, cum,
            durations, total_bin_cost, routing_cost, overall, penalty)


def _run_core(inst: Instance, pop: np.ndarray, mask: np.ndarray,
              lam: float, gamma: float):
    out = _decode_core(
        inst.W[1:], inst.C, inst.bin_cap, inst.bin_service, inst.bin_cost,
        float(inst.ccv), float(inst.tu), float(inst.Q), float(inst.TL),
        int(inst.n_vehicles), np.asarray(inst.work_days, dtype=np.int64),
        int(inst.n_days), float(lam), float(gamma), pop, mask)
    if not out[3]:
        raise ValueError(NO_BIN_FITS)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def accumulate(inst: Instance, mask: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Return w[i, t] = waste sitting at point i at the END of day t.
    See `_accumulate` for the model."""
    return _accumulate(inst.W[1:], mask,
                       np.asarray(inst.work_days, dtype=np.int64),
                       int(inst.n_days), int(max_iter))


def decode(inst: Instance, pop: np.ndarray, mask: np.ndarray,
           lam: float = 100.0, gamma: float = 1000.0) -> Solution:
    """Evaluate a candidate solution. `lam` and `gamma` are the fleet-size and
    shift-length penalty weights of Eq. (7)."""
    (w, wmax, bins, _, nr, route_day, ptr, stops, loads, cum, durations,
     bin_cost, routing_cost, overall, penalty) = _run_core(inst, pop, mask,
                                                           lam, gamma)

    stops_l, loads_l, cum_l = stops.tolist(), loads.tolist(), cum.tolist()
    routes = [Route(day=int(route_day[r]),
                    stops=stops_l[ptr[r]:ptr[r + 1]],
                    loads=loads_l[ptr[r]:ptr[r + 1]],
                    cum_load=cum_l[ptr[r]:ptr[r + 1]],
                    duration=float(durations[r]))
              for r in range(nr)]

    violations = []
    for day in range(7):
        n_routes = sum(1 for r in routes if r.day == day)
        if n_routes > inst.n_vehicles:
            violations.append(
                f"{DAY_NAMES[day]}: {n_routes} routes > fleet of {inst.n_vehicles}")
    for r in routes:
        if r.duration > inst.TL + 1e-9:
            violations.append(
                f"{DAY_NAMES[r.day]}: route {r.duration:.2f} min > TL {inst.TL:.2f}")

    return Solution(
        instance_name=inst.name, bins=bins, w=w, wmax=wmax, routes=routes,
        bin_cost=float(bin_cost), routing_cost=float(routing_cost),
        overall_cost=float(overall), feasible=(penalty == 0.0),
        penalty=float(penalty), violations=violations,
    )


def evaluate(inst: Instance, pop: np.ndarray, mask: np.ndarray,
             lam: float = 100.0, gamma: float = 1000.0) -> float:
    """Just the fitness -- what SA and GA call. Identical to
    decode(...).fitness, without building Route objects."""
    out = _run_core(inst, pop, mask, lam, gamma)
    return float(out[13] + out[14])
