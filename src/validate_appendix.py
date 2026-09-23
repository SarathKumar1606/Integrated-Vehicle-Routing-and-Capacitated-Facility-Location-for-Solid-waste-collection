"""
M8 - Validation Harness
=======================
Feeds the chromosome published in Appendix A (Table 16) of

    Gonzalez, Rossit, Frutos & Mendez (2025),
    "Modeling and solving an integrated periodic vehicle routing and
     capacitated facility location problem in the context of solid waste
     collection", Annals of Operations Research 350:979-1015

into OUR decoder, and checks the output against the paper's own Table 16
(bin choices, waste accumulation) and Table 17 (routes, loads, route times).

If every check passes, our implementation of the model is provably faithful,
and we can apply it to Chennai with confidence.

Run:  python src/validate_appendix.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from decoder import decode
from loader import load_instance

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Table 16, transcribed. Rows = collection points 1..12, columns = MON..SAT.
# ---------------------------------------------------------------------------
# p[i][d] = position of point i in day d's visiting order
POP = np.array([
    [8,  4,  4,  5,  7,  4],   # point 1
    [6,  4,  0,  6,  3,  3],   # point 2
    [5,  3,  2,  7,  2,  2],   # point 3
    [9,  2,  5,  8,  1,  8],   # point 4
    [10, 5,  6,  9,  5,  9],   # point 5
    [2,  8,  7,  3,  8,  6],   # point 6
    [1,  9,  8,  2,  9, 11],   # point 7
    [11, 6,  9, 10,  6,  0],   # point 8
    [7, 10,  3, 11,  4, 10],   # point 9
    [4, 11,  1,  1, 10,  1],   # point 10
    [0,  1, 10,  0, 11,  7],   # point 11
    [3,  7, 11,  4,  0,  5],   # point 12
])

# m[i][d] = 1 if point i is collected on day d
MASK = np.array([
    [0, 0, 1, 0, 0, 1],        # point 1
    [1, 1, 0, 0, 1, 1],        # point 2
    [1, 1, 1, 0, 1, 1],        # point 3
    [0, 1, 0, 0, 1, 1],        # point 4
    [0, 1, 0, 0, 1, 1],        # point 5
    [1, 0, 0, 1, 0, 1],        # point 6
    [1, 0, 0, 1, 0, 0],        # point 7
    [0, 1, 0, 0, 1, 0],        # point 8
    [1, 0, 1, 0, 1, 1],        # point 9
    [1, 0, 1, 1, 0, 1],        # point 10
    [0, 1, 0, 0, 0, 1],        # point 11
    [1, 1, 0, 1, 0, 1],        # point 12
])

# Bin combination chosen for each point (Table 16, column 2)
EXPECTED_BINS = np.array([7, 7, 2, 6, 6, 5, 7, 7, 4, 2, 5, 4])

# Accumulated waste w[i, t], MON..SAT (Table 16, w columns)
EXPECTED_W = np.array([
    [2.54, 3.81, 5.08, 1.27, 2.54, 3.81],
    [3.24, 1.62, 1.62, 3.24, 4.86, 1.62],
    [2.34, 1.17, 1.17, 1.17, 2.34, 1.17],
    [2.98, 4.47, 1.49, 2.98, 4.47, 1.49],
    [3.18, 4.77, 1.59, 3.18, 4.77, 1.59],
    [2.42, 1.21, 2.42, 3.63, 1.21, 2.42],
    [5.28, 1.32, 2.64, 3.96, 1.32, 2.64],
    [3.69, 4.92, 1.23, 2.46, 3.69, 1.23],
    [3.16, 1.58, 3.16, 1.58, 3.16, 1.58],
    [2.34, 1.17, 2.34, 1.17, 1.17, 2.34],
    [2.00, 3.00, 1.00, 2.00, 3.00, 4.00],
    [2.66, 1.33, 1.33, 2.66, 1.33, 2.66],
])

# ---------------------------------------------------------------------------
# Table 17: routes, their stop sequences and durations
# ---------------------------------------------------------------------------
EXPECTED_ROUTES = [
    (0, [7, 6, 12],            25.04),   # MON R1
    (0, [10, 3, 2, 9],         23.05),   # MON R2
    (1, [11, 4, 3, 2],         26.00),   # TUE R1
    (1, [5, 8, 12],            22.29),   # TUE R2
    (2, [10, 3, 9, 1],         25.80),   # WED R1
    (3, [10, 7, 6, 12],        26.00),   # THU R1
    (4, [4, 3, 2],             23.41),   # FRI R1
    (4, [9, 5, 8],             22.67),   # FRI R2
    (5, [10, 3, 2, 1, 12],     24.26),   # SAT R1
    (5, [6, 11, 4, 5, 9],      29.99),   # SAT R2
]


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=" * 68)
    print("VALIDATION AGAINST THE PAPER'S APPENDIX A (instance i.12.1)")
    print("=" * 68)

    inst = load_instance(ROOT / "data" / "12_1", "i.12.1")
    sol = decode(inst, POP, MASK)

    checks = []

    # --- 1. waste accumulation (Table 16) ---------------------------------
    got_w = sol.w[:, list(inst.work_days)]
    max_dev = float(np.abs(got_w - EXPECTED_W).max())
    checks.append(_check("waste accumulation w[i,t] matches Table 16",
                         max_dev < 5e-3, f"max deviation {max_dev:.4f} m3"))

    # --- 2. bin selection (Table 16) --------------------------------------
    same_bins = bool((sol.bins == EXPECTED_BINS).all())
    checks.append(_check("bin combination per point matches Table 16",
                         same_bins,
                         "" if same_bins else f"got {sol.bins.tolist()}"))

    # --- 3. routes and durations (Table 17) -------------------------------
    got = [(r.day, [s + 1 for s in r.stops], round(r.duration, 2))
           for r in sol.routes]
    n_ok = _check("number of routes matches Table 17",
                  len(got) == len(EXPECTED_ROUTES),
                  f"got {len(got)}, expected {len(EXPECTED_ROUTES)}")
    checks.append(n_ok)

    if n_ok:
        seq_ok, time_ok, worst = True, True, 0.0
        for (gd, gs, gt), (ed, es, et) in zip(got, EXPECTED_ROUTES):
            if gd != ed or gs != es:
                seq_ok = False
                print(f"        sequence mismatch: got {gs}, expected {es}")
            worst = max(worst, abs(gt - et))
            if abs(gt - et) > 5e-3:
                time_ok = False
                print(f"        duration mismatch: got {gt}, expected {et}")
        checks.append(_check("route stop sequences match Table 17", seq_ok))
        checks.append(_check("route durations match Table 17", time_ok,
                             f"max deviation {worst:.4f} min"))

    # --- 4. capacity and feasibility --------------------------------------
    worst_load = max((r.total_load for r in sol.routes), default=0.0)
    checks.append(_check("no route exceeds vehicle capacity",
                         worst_load <= inst.Q + 1e-9,
                         f"heaviest route {worst_load:.2f} / {inst.Q:.1f} m3"))
    checks.append(_check("solution is feasible (fleet + shift limits)",
                         sol.feasible))

    # --- summary ----------------------------------------------------------
    print("-" * 68)
    print(sol.report())
    print("-" * 68)
    passed = all(checks)
    print(f"RESULT: {sum(checks)}/{len(checks)} checks passed"
          f"  ->  {'VALIDATED' if passed else 'MISMATCH'}")
    print("=" * 68)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
