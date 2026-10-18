"""
Check the browser-solver exports against the live Python code, so they
cannot silently drift from the solver.

  ui/data/solver_core.json  matrices and parameters equal load_instance()
                            and penalty_weights(); every parity case
                            re-decodes to EXACTLY the stored costs, penalty,
                            bins, stops and durations (float ==); the
                            validation block reproduces Appendix A.
  ui/data/geometry.json     its nodes are the instance's nodes, all 462
                            ordered pairs are present, and every polyline
                            starts and ends near the right nodes.

Run:  python scripts/verify_js_fixtures.py
Re-export with src/export_solver_core.py / src/export_geometry.py if it fails.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                       # noqa: E402

import validate_appendix as va                           # noqa: E402
from decoder import decode                               # noqa: E402
from loader import load_instance                         # noqa: E402
from optimizers import penalty_weights                   # noqa: E402

DATA = ROOT / "ui" / "data"
FOLDERS = {"chennai_guindy": ROOT / "data" / "chennai_guindy",
           "chennai_guindy_peak": ROOT / "data" / "chennai_guindy_peak",
           "i.12.1": ROOT / "data" / "12_1"}
FAILS: list[str] = []


def check(cond: bool, msg: str) -> bool:
    if not cond:
        FAILS.append(msg)
        if len(FAILS) <= 15:
            print(f"  FAIL: {msg}")
    return cond


def same(a, b) -> bool:
    """Exact equality for nested lists of numbers (no tolerance)."""
    return np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


# ---------------------------------------------------------------------------
def check_instance(name: str, blk: dict, with_points: bool = True):
    inst = load_instance(FOLDERS[name], name)
    lam, gamma = penalty_weights(inst)
    check(same(blk["C"], inst.C), f"{name}: C differs from times.txt")
    check(same(blk["W"], inst.W), f"{name}: W differs")
    for key, want in (("Q", inst.Q), ("n_vehicles", inst.n_vehicles),
                      ("TL", inst.TL), ("ccv", inst.ccv), ("tu", inst.tu),
                      ("n_days", inst.n_days), ("lam", lam), ("gamma", gamma),
                      ("cost_scale", inst.cost_scale),
                      ("currency", inst.currency),
                      ("currency_symbol", inst.currency_symbol),
                      ("fx_rate_from_usd", inst.fx_rate)):
        check(blk[key] == want, f"{name}: {key} {blk[key]!r} != {want!r}")
    check(blk["work_days"] == list(inst.work_days)
          and blk["rest_days"] == list(inst.rest_days), f"{name}: days differ")
    check(same([b["capacity_m3"] for b in blk["bins"]], inst.bin_cap)
          and same([b["service_min"] for b in blk["bins"]], inst.bin_service)
          and same([b["weekly_cost"] for b in blk["bins"]], inst.bin_cost),
          f"{name}: bins differ")
    if with_points:
        check(same([p["lat"] for p in blk["points"]], inst.lat)
              and same([p["lon"] for p in blk["points"]], inst.lon),
              f"{name}: point coordinates differ")
    return inst, lam, gamma


def check_case(inst, lam, gamma, case: dict, tag: str) -> None:
    sol = decode(inst, np.array(case["pop"]), np.array(case["mask"]), lam, gamma)
    ok = (sol.bins.tolist() == case["bins"]
          and float(sol.bin_cost) == case["bin_cost"]
          and float(sol.routing_cost) == case["routing_cost"]
          and float(sol.overall_cost) == case["overall_cost"]
          and float(sol.penalty) == case["penalty"]
          and float(sol.fitness) == case["fitness"]
          and bool(sol.feasible) == case["feasible"]
          and [(r.day, list(r.stops), float(r.duration)) for r in sol.routes]
          == [(r["day"], r["stops"], r["duration"]) for r in case["routes"]])
    check(ok, f"{tag}: Python decode no longer matches the stored result")


def verify_solver_core() -> None:
    path = DATA / "solver_core.json"
    if not check(path.exists(), f"{path} missing"):
        return
    doc = json.loads(path.read_text(encoding="utf-8"))

    for name, blk in doc["instances"].items():
        inst, lam, gamma = check_instance(name, blk)
        for key in ("parity", "parity_feasible"):
            cases = doc.get(key, {}).get(name, [])
            for k, case in enumerate(cases):
                check_case(inst, lam, gamma, case, f"{key} {name} #{k}")
            print(f"  {name:<20} {key:<16} {len(cases):3} cases re-decoded")

    v = doc["validation"]
    inst, lam, gamma = check_instance("i.12.1", v, with_points=False)
    check(v["pop"] == va.POP.tolist() and v["mask"] == va.MASK.tolist(),
          "validation: pop/mask differ from validate_appendix.py")
    check_case(inst, lam, gamma, {"pop": v["pop"], "mask": v["mask"],
                                  **v["python_exact"]}, "validation")
    sol = decode(inst, va.POP, va.MASK, lam, gamma)
    pe = v["paper_expected"]
    got_w = sol.w[:, list(inst.work_days)]
    check(np.abs(got_w - np.array(pe["w_work_days"])).max() < pe["tolerance"],
          "validation: waste accumulation differs from Table 16")
    check(sol.bins.tolist() == pe["bins"], "validation: bins differ from Table 16")
    routes = [(r.day, [s + 1 for s in r.stops], r.duration) for r in sol.routes]
    check(len(routes) == len(pe["routes"]) and all(
        d == e["day"] and s == e["stops_1based"]
        and abs(t - e["duration"]) < pe["tolerance"]
        for (d, s, t), e in zip(routes, pe["routes"])),
        "validation: routes differ from Table 17")
    print(f"  validation (i.12.1)  Appendix A reproduced: overall "
          f"{sol.overall_cost:.2f} US$, {len(routes)} routes")


# ---------------------------------------------------------------------------
def decode_polyline(s: str, precision: int = 5) -> list[tuple[float, float]]:
    """Google encoded polyline -> [(lat, lon), ...]."""
    coords, idx, lat, lon, f = [], 0, 0, 0, 10 ** precision
    while idx < len(s):
        for which in (0, 1):
            shift = result = 0
            while True:
                b = ord(s[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if which == 0:
                lat += delta
            else:
                lon += delta
        coords.append((lat / f, lon / f))
    return coords


def km(a, b) -> float:
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    h = (math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2)
         * math.sin(math.radians(lo2 - lo1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def verify_geometry(snap_km: float = 1.0) -> None:
    path = DATA / "geometry.json"
    if not check(path.exists(), f"{path} missing"):
        return
    doc = json.loads(path.read_text(encoding="utf-8"))
    inst = load_instance(FOLDERS["chennai_guindy"])
    nodes = doc["nodes"]
    check(same([n["lat"] for n in nodes], inst.lat)
          and same([n["lon"] for n in nodes], inst.lon),
          "geometry: nodes differ from the instance")
    n = inst.n_points + 1
    want = {f"{i}-{j}" for i in range(n) for j in range(n) if i != j}
    pairs = doc["pairs"]
    check(set(pairs) == want, f"geometry: {len(want ^ set(pairs))} keys wrong")
    ok = 0
    for key, v in pairs.items():
        if v is None:
            continue
        i, j = map(int, key.split("-"))
        pts = decode_polyline(v["polyline"], doc.get("polyline_precision", 5))
        good = (len(pts) >= 2 and v["distance_km"] > 0 and v["duration_min"] > 0
                and km(pts[0], (nodes[i]["lat"], nodes[i]["lon"])) < snap_km
                and km(pts[-1], (nodes[j]["lat"], nodes[j]["lon"])) < snap_km)
        ok += check(good, f"geometry {key}: polyline/distance/endpoints wrong")
    nulls = sum(v is None for v in pairs.values())
    check(doc["n_ok"] == len(pairs) - nulls, "geometry: n_ok inconsistent")
    print(f"  geometry             {ok}/{len(want)} legs valid, {nulls} null "
          f"(straight-line fallback)")


def main() -> int:
    print("solver_core.json")
    verify_solver_core()
    print("geometry.json")
    verify_geometry()
    print(f"RESULT: {'FIXTURES MATCH THE PYTHON SOLVER' if not FAILS else f'{len(FAILS)} FAILURES'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
