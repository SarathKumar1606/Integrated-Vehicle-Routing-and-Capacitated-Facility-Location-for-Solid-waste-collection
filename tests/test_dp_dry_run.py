"""
The toy instance of docs/dp_dry_run.md, run through the real code.

Every figure in the worked example -- waste accumulation, bin choice, the
greedy routes, each Held-Karp entry, the partition table and the final DP
routes and costs -- is asserted here, so the document and the code cannot
drift apart.

Run:  python tests/test_dp_dry_run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from decoder import decode                                      # noqa: E402
from dp_router import (dp_day_routes, held_karp_all_subsets,    # noqa: E402
                       polish_with_report)
from loader import Instance                                     # noqa: E402

NAMES = ["A", "B", "C", "D"]
# travel times in minutes; row/column 0 is the depot
C = np.array([[0, 5, 7, 6, 8],
              [5, 0, 3, 4, 6],
              [7, 3, 0, 5, 4],
              [6, 4, 5, 0, 3],
              [8, 6, 4, 3, 0]], dtype=float)
# collection days (columns: day 1, day 2; day 3 is the rest day)
MASK = np.array([[0, 1],          # A: day 2 only
                 [1, 1],          # B: both days
                 [1, 1],          # C: both days
                 [1, 1]])         # D: both days
# visiting order each day for the greedy decoder: A, B, C, D
POP = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])


def toy_instance() -> Instance:
    return Instance(
        name="toy", n_points=4, ids=np.arange(5),
        lon=np.zeros(5), lat=np.zeros(5),
        W=np.array([0.0, 1.0, 1.5, 0.8, 1.2]), C=C,
        bin_cap=np.array([2.0, 3.5]), bin_service=np.array([1.0, 1.5]),
        bin_cost=np.array([100.0, 160.0]),
        Q=4.0, n_vehicles=2, TL=30.0, ccv=50.0, tu=5.0,
        n_days=3, rest_days=(2,), currency="INR", currency_symbol="₹")


def mask_of(day_pts, members: str) -> int:
    return sum(1 << j for j, p in enumerate(day_pts) if NAMES[p] in members)


def main() -> int:
    inst = toy_instance()
    sol = decode(inst, POP, MASK, 100.0, 1000.0)
    fails = []

    def eq(got, want, what):
        if not np.isclose(got, want, atol=1e-9):
            fails.append(f"{what}: got {got}, document says {want}")

    # --- section 2: waste accumulation and bins ---------------------------
    w_doc = {"A": [2.0, 3.0, 1.0], "B": [3.0, 1.5, 1.5],
             "C": [1.6, 0.8, 0.8], "D": [2.4, 1.2, 1.2]}
    for i, n in enumerate(NAMES):
        for t in range(3):
            eq(sol.w[i, t], w_doc[n][t], f"w[{n}, day {t + 1}]")
    if sol.bins.tolist() != [1, 1, 0, 1]:
        fails.append(f"bins {sol.bins.tolist()} != [1, 1, 0, 1]")
    eq(sol.bin_cost, 580.0, "bin cost")

    # --- section 3: greedy routes -----------------------------------------
    greedy = [(r.day, "".join(NAMES[s] for s in r.stops), r.duration)
              for r in sol.routes]
    if [(d, s) for d, s, _ in greedy] != [(0, "B"), (0, "CD"), (1, "A"), (1, "BCD")]:
        fails.append(f"greedy routes {greedy}")
    for (_, _, got), want in zip(greedy, [20.5, 24.5, 16.5, 32.0]):
        eq(got, want, "greedy route duration")
    eq(sol.routing_cost, 50.0 * 93.5, "greedy routing cost")
    eq(sol.penalty, 1000.0 * 2.0, "greedy penalty (BCD 2 min over TL)")

    # --- section 4: Held-Karp on day 2 ------------------------------------
    pts = np.array([0, 1, 2, 3], dtype=np.int64)
    svc = np.array([1.5, 1.5, 1.0, 1.5])
    closed, dp, parent, end = held_karp_all_subsets(pts, C, svc, 5.0)
    dp_doc = {"A": {"A": 5}, "B": {"B": 7}, "C": {"C": 6}, "D": {"D": 8},
              "AB": {"A": 10, "B": 8}, "AC": {"A": 10, "C": 9},
              "AD": {"A": 14, "D": 11}, "BC": {"B": 11, "C": 12},
              "BD": {"B": 12, "D": 11}, "CD": {"C": 11, "D": 9},
              "ABC": {"A": 14, "B": 13, "C": 13}, "ABD": {"A": 15, "B": 15, "D": 12},
              "ACD": {"A": 15, "C": 14, "D": 12}, "BCD": {"B": 13, "C": 14, "D": 15},
              "ABCD": {"A": 16, "B": 16, "C": 15, "D": 16}}
    closed_doc = {"A": 16.5, "B": 20.5, "C": 18.0, "D": 22.5, "AB": 23.0,
                  "AC": 22.5, "AD": 27.0, "BC": 25.5, "BD": 27.0, "CD": 24.5,
                  "ABC": 28.0, "ABD": 29.5, "ACD": 29.0, "BCD": 29.0,
                  "ABCD": 31.5}
    for members, row in dp_doc.items():
        S = mask_of(pts, members)
        for j_name, want in row.items():
            eq(dp[S, NAMES.index(j_name)], want, f"dp[{members}][{j_name}]")
        eq(closed[S], closed_doc[members], f"closed({members})")

    # --- section 5: partition DP on day 2 ---------------------------------
    ok, total, nr, ptr, stops = dp_day_routes(
        pts, np.array([3.0, 1.5, 0.8, 1.2]), sol.bins, C, inst.bin_service,
        4.0, 30.0, 2, 5.0)
    routes = ["".join(NAMES[s] for s in stops[ptr[r]:ptr[r + 1]]) for r in range(nr)]
    if not ok or routes != ["A", "CDB"]:
        fails.append(f"day 2 DP routes {routes}")
    eq(total, 45.5, "day 2 DP total")

    # the partition table of section 5: best total with <= 1 and <= 2 routes
    # for every subset S of day 2 (inf = no admissible way)
    inf = np.inf
    f_doc = {"A": (16.5, 16.5), "B": (20.5, 20.5), "C": (18.0, 18.0),
             "D": (22.5, 22.5), "AB": (inf, 37.0), "AC": (22.5, 22.5),
             "AD": (inf, 39.0), "BC": (25.5, 25.5), "BD": (27.0, 27.0),
             "CD": (24.5, 24.5), "ABC": (inf, 42.0), "ABD": (inf, 43.5),
             "ACD": (inf, 41.0), "BCD": (29.0, 29.0), "ABCD": (inf, 45.5)}
    loads = {"A": 3.0, "B": 1.5, "C": 0.8, "D": 1.2}
    for members, (k1, k2) in f_doc.items():
        sub = np.array([NAMES.index(m) for m in members], dtype=np.int64)
        wl = np.array([loads[m] for m in members])
        for K, want in ((1, k1), (2, k2)):
            ok_s, tot_s, *_ = dp_day_routes(sub, wl, sol.bins, C, inst.bin_service,
                                            4.0, 30.0, K, 5.0)
            got = tot_s if ok_s else inf
            if not (got == want or np.isclose(got, want)):
                fails.append(f"best({members}, <= {K} routes): got {got}, "
                             f"document says {want}")

    # --- section 6: the polished week --------------------------------------
    pol, rep = polish_with_report(inst, sol, 100.0, 1000.0)
    eq(pol.routing_cost, 50.0 * 90.5, "polished routing cost")
    eq(pol.overall_cost, 580.0 + 4525.0, "polished overall cost")
    if not pol.feasible or [d.status for d in rep.days] != ["unchanged", "improved"]:
        fails.append(f"polish: feasible {pol.feasible}, "
                     f"{[d.status for d in rep.days]}")

    for f in fails:
        print("  FAIL:", f)
    print(f"RESULT: {'dry run reproduced exactly' if not fails else f'{len(fails)} mismatches'}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
