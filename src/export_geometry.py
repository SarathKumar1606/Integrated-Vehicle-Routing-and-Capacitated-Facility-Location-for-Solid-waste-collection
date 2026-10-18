"""
Export ui/data/geometry.json: OSRM road geometry for every ordered pair of
the 22 Chennai nodes (depot + 21 points), so a route computed in the browser
can be drawn on real streets by stitching its legs.

One request at a time to the public OSRM demo server, a pause between
requests, retry with backoff (instance_builder.get_json), and every completed
pair appended to a cache file as it arrives, so an interrupted run resumes
where it stopped. Pairs that still fail are stored as null.

Run:  python src/export_geometry.py            # resumes from the cache
"""

from __future__ import annotations

import json
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from instance_builder import get_json
from loader import load_instance

ROOT = Path(__file__).parent.parent
OUT = ROOT / "ui" / "data" / "geometry.json"
CACHE = ROOT / "results" / "cache" / "osrm_pairs.jsonl"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving/"
PAUSE_S = 1.0

COMMENT = (
    "Road geometry for every ordered pair (i, j), i != j, of the 22 Chennai "
    "nodes; node 0 is the Perungudi depot and node i is collection point i "
    "(the same indexing as the travel-time matrix C in solver_core.json). "
    "Keys are \"i-j\". Each value has the leg's encoded polyline (Google "
    "polyline format, precision 5, OSRM overview=simplified), its road "
    "distance in km and OSRM's free-flow duration in minutes; null means "
    "OSRM failed for that pair and the UI should draw a straight line and "
    "say so. Use these to DRAW routes only: route durations and costs come "
    "from C (which is ×1.5 in the peak scenario). A route drawn by stitching "
    "pairwise legs can differ slightly from OSRM's own multi-waypoint route "
    "for the same stops, since each leg is optimised independently (e.g. "
    "where to approach or leave a stop). Exported by src/export_geometry.py.")


def nodes() -> list[dict]:
    a = load_instance(ROOT / "data" / "chennai_guindy")
    b = load_instance(ROOT / "data" / "chennai_guindy_peak")
    if not ((a.lat == b.lat).all() and (a.lon == b.lon).all()):
        raise SystemExit("free-flow and peak instances have different nodes")
    names = {}
    with (ROOT / "data" / "chennai_guindy" / "points.csv").open(encoding="utf-8") as fh:
        import csv
        names = {int(r["id"]): r["name"] for r in csv.DictReader(fh)}
    return [{"id": i, "name": names.get(i, str(i)),
             "lat": float(a.lat[i]), "lon": float(a.lon[i])}
            for i in range(a.n_points + 1)]


def load_cache() -> dict:
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["key"]] = rec["value"]
    return done


def fetch_leg(a: dict, b: dict) -> dict:
    url = (f"{OSRM_ROUTE_URL}{a['lon']:.6f},{a['lat']:.6f};"
           f"{b['lon']:.6f},{b['lat']:.6f}"
           "?overview=simplified&geometries=polyline")
    data = get_json(url, timeout=30)
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM returned {data.get('code')}")
    r = data["routes"][0]
    return {"polyline": r["geometry"],
            "distance_km": r["distance"] / 1000.0,
            "duration_min": r["duration"] / 60.0}


def main() -> None:
    pts = nodes()
    keys = [(i, j) for i in range(len(pts)) for j in range(len(pts)) if i != j]
    done = load_cache()
    todo = [(i, j) for i, j in keys if f"{i}-{j}" not in done]
    print(f"{len(keys)} ordered pairs: {len(keys) - len(todo)} cached, "
          f"{len(todo)} to fetch")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    failed = {}
    t0 = time.time()
    with CACHE.open("a", encoding="utf-8") as cache:
        for n, (i, j) in enumerate(todo, 1):
            key = f"{i}-{j}"
            try:
                value = fetch_leg(pts[i], pts[j])
            except (urllib.error.URLError, OSError, RuntimeError, TimeoutError,
                    KeyError, ValueError) as exc:
                failed[key] = str(exc)
                print(f"  ! {key} failed after retries: {exc}")
            else:
                done[key] = value
                cache.write(json.dumps({"key": key, "value": value}) + "\n")
                cache.flush()
            if n % 50 == 0:
                print(f"  {n}/{len(todo)} fetched ({time.time() - t0:.0f} s)")
            time.sleep(PAUSE_S)

    pairs = {f"{i}-{j}": done.get(f"{i}-{j}") for i, j in keys}
    ok = sum(v is not None for v in pairs.values())
    doc = {
        "_comment": COMMENT,
        "source": OSRM_ROUTE_URL + "{lon,lat};{lon,lat}"
                  "?overview=simplified&geometries=polyline",
        "polyline_precision": 5,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nodes": pts,
        "n_pairs": len(keys),
        "n_ok": ok,
        "failed": sorted(k for k, v in pairs.items() if v is None),
        "pairs": pairs,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
                   + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {ok}/{len(keys)} pairs ok, "
          f"{OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
