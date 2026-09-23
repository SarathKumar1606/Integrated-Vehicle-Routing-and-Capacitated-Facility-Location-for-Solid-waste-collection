"""
Regression test: the Numba decoder must reproduce the original pure-Python
decoder (tests/reference_decoder.py) BIT FOR BIT -- every w[i, t], bin, stop,
load, duration, cost, penalty and violation message, on thousands of random
solutions, including infeasible ones and ones where no bin fits.

Run:  python tests/test_decoder_equivalence.py [cases_per_instance] [extra_data_dir]

Every instance folder under data/ is tested, plus any under extra_data_dir
(e.g. a checkout of the full Gonzalez et al. dataset).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import decoder as fast                     # noqa: E402
import reference_decoder as ref            # noqa: E402
from loader import load_instance           # noqa: E402


def _bits(x) -> str:
    return float(x).hex()


def _fingerprint(sol) -> tuple:
    return (
        np.asarray(sol.bins, dtype=np.int64).tobytes(),
        np.asarray(sol.w, dtype=np.float64).tobytes(),
        np.asarray(sol.wmax, dtype=np.float64).tobytes(),
        [(int(r.day), [int(s) for s in r.stops], [_bits(v) for v in r.loads],
          [_bits(v) for v in r.cum_load], _bits(r.duration)) for r in sol.routes],
        _bits(sol.bin_cost), _bits(sol.routing_cost), _bits(sol.overall_cost),
        bool(sol.feasible), _bits(sol.penalty), list(sol.violations),
        _bits(sol.fitness),
    )


def _outcome(module, inst, pop, mask, lam, gamma):
    try:
        return _fingerprint(module.decode(inst, pop, mask, lam, gamma))
    except ValueError as exc:
        return ("ValueError", str(exc))


def _cases(inst, n, rng):
    nI, nD = inst.n_points, inst.n_work_days
    for k in range(n):
        pop = np.column_stack([rng.permutation(nI) for _ in range(nD)])
        kind = k % 5
        if kind == 0:                                   # sparse
            mask = (rng.random((nI, nD)) < 0.3).astype(np.int8)
        elif kind == 1:                                 # dense
            mask = (rng.random((nI, nD)) < 0.9).astype(np.int8)
        elif kind == 2:                                 # every day
            mask = np.ones((nI, nD), dtype=np.int64)
        else:                                           # mixed, int64 dtype
            mask = (rng.random((nI, nD)) < rng.random()).astype(np.int64)
        if kind == 4:                                   # tie-breaking in sort
            pop = rng.integers(0, max(nI // 2, 1), size=(nI, nD))
        lam, gamma = (100.0, 1000.0) if k % 2 else (rng.random() * 500,
                                                    rng.random() * 5000)
        yield pop, mask, lam, gamma


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rng = np.random.default_rng(20250923)
    failures = 0
    roots = [ROOT / "data"] + [Path(a) for a in sys.argv[2:]]
    folders = sorted(p for r in roots for p in r.iterdir()
                     if (p / "waste.txt").exists())
    for folder in folders:
        try:
            inst = load_instance(folder)
        except ValueError as exc:
            print(f"  {folder.name:<16} skipped: loader cannot read it ({exc})")
            continue
        checked = errors = 0
        for pop, mask, lam, gamma in _cases(inst, n, rng):
            want = _outcome(ref, inst, pop, mask, lam, gamma)
            got = _outcome(fast, inst, pop, mask, lam, gamma)
            acc_ok = (ref.accumulate(inst, mask).tobytes()
                      == fast.accumulate(inst, mask).tobytes())
            eval_ok = True
            if want[0] != "ValueError":
                eval_ok = _bits(fast.evaluate(inst, pop, mask, lam, gamma)) == want[-1]
            else:
                errors += 1
            if want != got or not acc_ok or not eval_ok:
                failures += 1
                if failures <= 5:
                    print(f"  MISMATCH on {folder.name}: decode={want == got} "
                          f"accumulate={acc_ok} evaluate={eval_ok}")
            checked += 1
        print(f"  {folder.name:<16} {checked} cases ({errors} no-bin-fits)")
    print(f"RESULT: {'BIT-FOR-BIT IDENTICAL' if failures == 0 else f'{failures} MISMATCHES'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
