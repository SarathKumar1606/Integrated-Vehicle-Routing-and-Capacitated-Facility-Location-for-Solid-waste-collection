"""
M7 - Exporter
=============
Turns the best solution found for an instance into one JSON file that the
Leaflet map (M9) can draw directly:

  * every collection point: coordinates, name, chosen bin combination and its
    capacity, and the fill level of that bin on each day of the week;
  * every day's routes: ordered stops with cumulative load and elapsed time,
    route duration broken into travel / service / unloading, and the real
    road geometry of the route from the OSRM route service;
  * the headline costs, exactly as the decoder computes them.

Where the solution comes from
-----------------------------
runner.py saves each run's best chromosome to results/solutions/. The
exporter decodes all of them for the instance and keeps the best: feasible
first, then lowest fitness. `--solution FILE.npz` exports a specific one.

Road geometry
-------------
Routes are fetched from OSRM with retry and backoff, one request per route,
paced for the public demo server. If OSRM cannot be reached, that route falls
back to straight lines between stops, and `geometry_source` records this per
route so the map (and the report) can never mistake one for the other.

Usage:
    python src/exporter.py --instance data/chennai_guindy
    python src/exporter.py --instance data/12_1 --name i.12.1 --offline
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from decoder import Solution, decode
from instance_builder import osrm_get
from loader import DAY_NAMES, Instance, load_instance

ROOT = Path(__file__).parent.parent
SOLUTIONS_DIR = ROOT / "results" / "solutions"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving/"
OSRM_PAUSE_S = 1.0          # be polite to the public demo server


# ---------------------------------------------------------------------------
# Choosing the solution
# ---------------------------------------------------------------------------
def best_saved_solution(inst: Instance, name: str) -> tuple[Path, Solution]:
    """Decode every saved chromosome for this instance; keep the best."""
    files = sorted(SOLUTIONS_DIR.glob(f"{name}_*.npz"))
    if not files:
        raise SystemExit(
            f"No saved solutions for '{name}' in {SOLUTIONS_DIR}.\n"
            f"Run first:  python src/runner.py --instance <folder> --name {name}")
    best = None
    for f in files:
        with np.load(f) as z:
            sol = decode(inst, z["pop"], z["mask"])
        key = (not sol.feasible, sol.fitness)
        if best is None or key < best[0]:
            best = (key, f, sol)
    return best[1], best[2]


def load_solution_file(inst: Instance, path: Path) -> Solution:
    with np.load(path) as z:
        return decode(inst, z["pop"], z["mask"])


# ---------------------------------------------------------------------------
# Point names
# ---------------------------------------------------------------------------
def point_names(folder: Path, n_nodes: int) -> list[str]:
    """Names from the instance's points.csv (written by instance_builder.py);
    generic labels for instances that have none, such as the paper's."""
    names = ["Depot"] + [f"Point {i}" for i in range(1, n_nodes)]
    csv_path = folder / "points.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                idx = int(row["id"])
                if 0 <= idx < n_nodes:
                    names[idx] = row["name"]
    return names


# ---------------------------------------------------------------------------
# Road geometry
# ---------------------------------------------------------------------------
def osrm_route(lonlats: list[tuple[float, float]]) -> dict:
    """Road geometry for a sequence of (lon, lat) waypoints, as GeoJSON."""
    locs = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in lonlats)
    data = osrm_get(f"{OSRM_ROUTE_URL}{locs}?overview=full&geometries=geojson")
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM returned {data.get('code')}")
    r = data["routes"][0]
    return {"geometry": r["geometry"],
            "road_distance_km": round(r["distance"] / 1000.0, 3),
            "osrm_duration_min": round(r["duration"] / 60.0, 2)}


def straight_lines(lonlats: list[tuple[float, float]]) -> dict:
    return {"geometry": {"type": "LineString",
                         "coordinates": [list(p) for p in lonlats]},
            "road_distance_km": None, "osrm_duration_min": None}


# ---------------------------------------------------------------------------
# Building the JSON
# ---------------------------------------------------------------------------
def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def build_export(inst: Instance, sol: Solution, folder: Path,
                 source: str, offline: bool = False) -> dict:
    names = point_names(folder, inst.n_points + 1)
    lonlat = lambda node: (float(inst.lon[node]), float(inst.lat[node]))  # noqa: E731

    # --- collection points ------------------------------------------------
    collected_on = {i: [] for i in range(inst.n_points)}
    for r in sol.routes:
        for s in r.stops:
            collected_on[s].append(r.day)

    points = []
    for i in range(inst.n_points):
        b = int(sol.bins[i])
        cap = float(inst.bin_cap[b])
        days = sorted(collected_on[i])
        points.append({
            "id": i + 1,                       # matrix index; 0 is the depot
            "name": names[i + 1],
            "lat": float(inst.lat[i + 1]),
            "lon": float(inst.lon[i + 1]),
            "waste_m3_per_day": float(inst.W[i + 1]),
            "bin": {"combination": b, "capacity_m3": cap,
                    "service_min": float(inst.bin_service[b]),
                    "weekly_cost": float(inst.bin_cost[b])},
            "visits_per_week": len(days),
            "collection_days": [DAY_NAMES[d] for d in days],
            "peak_waste_m3": _r(sol.wmax[i]),
            # waste in the bin at the end of each day, before any collection
            "fill": [{"day": DAY_NAMES[t],
                      "waste_m3": _r(sol.w[i, t]),
                      "fill_ratio": _r(sol.w[i, t] / cap),
                      "collected": t in days}
                     for t in range(inst.n_days)],
        })

    # --- routes -----------------------------------------------------------
    days = []
    geometry_cache: dict[tuple, dict] = {}
    n_fetched = 0
    for day in range(inst.n_days):
        day_routes = []
        for k, r in enumerate(sol.routes_on(day), start=1):
            nodes = [0] + [s + 1 for s in r.stops] + [0]

            # elapsed time, following Eq. (1) leg by leg
            stops, t, travel, service = [], 0.0, 0.0, 0.0
            for pos, s in enumerate(r.stops):
                leg = float(inst.C[nodes[pos], nodes[pos + 1]])
                travel += leg
                t += leg
                arrive = t
                svc = float(inst.bin_service[sol.bins[s]])
                service += svc
                t += svc
                stops.append({
                    "order": pos + 1,
                    "point_id": s + 1,
                    "name": names[s + 1],
                    "lat": float(inst.lat[s + 1]),
                    "lon": float(inst.lon[s + 1]),
                    "load_m3": _r(r.loads[pos]),
                    "cum_load_m3": _r(r.cum_load[pos]),
                    "arrive_min": _r(arrive, 2),
                    "depart_min": _r(t, 2),
                })
            travel += float(inst.C[nodes[-2], 0])
            if abs(travel + service + inst.tu - r.duration) > 1e-6:
                raise AssertionError("elapsed-time breakdown disagrees with "
                                     "the decoder's route duration")

            key = tuple(nodes)
            if key not in geometry_cache:
                pts = [lonlat(n) for n in nodes]
                geo, geo_src = None, "straight_line"
                if not offline:
                    if n_fetched:
                        time.sleep(OSRM_PAUSE_S)
                    try:
                        geo, geo_src = osrm_route(pts), "osrm"
                    except (urllib.error.URLError, OSError, RuntimeError,
                            TimeoutError, KeyError, ValueError) as exc:
                        print(f"  ! OSRM route failed for {DAY_NAMES[day]} "
                              f"R{k} ({exc}); drawing straight lines.")
                    n_fetched += 1
                geometry_cache[key] = {**(geo or straight_lines(pts)),
                                       "geometry_source": geo_src}

            day_routes.append({
                "route": k,
                "stops": stops,
                "n_stops": len(stops),
                "total_load_m3": _r(r.total_load),
                "vehicle_capacity_m3": float(inst.Q),
                "load_ratio": _r(r.total_load / inst.Q),
                "duration_min": _r(r.duration, 2),
                "travel_min": _r(travel, 2),
                "service_min": _r(service, 2),
                "unload_min": float(inst.tu),
                "within_shift": bool(r.duration <= inst.TL + 1e-9),
                **geometry_cache[key],
            })
        days.append({"day": DAY_NAMES[day],
                     "rest_day": day in inst.rest_days,
                     "n_routes": len(day_routes),
                     "duration_min": _r(sum(x["duration_min"] for x in day_routes), 2),
                     "routes": day_routes})

    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    return {
        "instance": {
            "name": inst.name,
            "n_points": inst.n_points,
            "travel_time_source": meta.get("travel_time_source", "dataset"),
            "vehicle_capacity_m3": float(inst.Q),
            "fleet_size": int(inst.n_vehicles),
            "shift_length_min": float(inst.TL),
            "cost_per_min": float(inst.ccv),
            "unload_min": float(inst.tu),
            "work_days": [DAY_NAMES[d] for d in inst.work_days],
            "rest_days": [DAY_NAMES[d] for d in inst.rest_days],
        },
        "depot": {"id": 0, "name": names[0],
                  "lat": float(inst.lat[0]), "lon": float(inst.lon[0])},
        "points": points,
        "days": days,
        "costs": {
            "currency": "US$",
            "bin_cost": _r(sol.bin_cost),
            "routing_cost": _r(sol.routing_cost),
            "overall_cost": _r(sol.overall_cost),
            "penalty": _r(sol.penalty),
            "fitness": _r(sol.fitness),
            "feasible": bool(sol.feasible),
            "violations": list(sol.violations),
            "total_route_minutes": _r(sum(r.duration for r in sol.routes), 2),
            "n_routes": len(sol.routes),
        },
        "provenance": {
            "solution_file": source,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "geometry_sources": sorted({g["geometry_source"]
                                        for g in geometry_cache.values()}),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True, help="instance folder")
    ap.add_argument("--name", default=None,
                    help="instance name used by runner.py (default: folder name)")
    ap.add_argument("--solution", default=None,
                    help="a specific results/solutions/*.npz to export")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: ui/data/<name>.json)")
    ap.add_argument("--offline", action="store_true",
                    help="skip OSRM; draw straight lines between stops")
    args = ap.parse_args()

    folder = Path(args.instance)
    name = args.name or folder.name
    inst = load_instance(folder, name)

    if args.solution:
        path = Path(args.solution)
        sol = load_solution_file(inst, path)
    else:
        path, sol = best_saved_solution(inst, name)
    print(f"  exporting {path.name}: overall {sol.overall_cost:.2f} US$, "
          f"feasible {sol.feasible}, {len(sol.routes)} routes")

    try:
        source = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        source = path.as_posix()
    data = build_export(inst, sol, folder, source, offline=args.offline)

    out = Path(args.out) if args.out else ROOT / "ui" / "data" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"  wrote {out}")
    print(f"  road geometry from: {', '.join(data['provenance']['geometry_sources'])}")


if __name__ == "__main__":
    main()
