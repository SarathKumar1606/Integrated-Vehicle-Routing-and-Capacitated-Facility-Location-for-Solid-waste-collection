"""
M1 + M2 - Chennai Instance Builder
==================================
Builds a collection-point instance for the CEG Guindy corridor in the exact
file format the paper uses, so the same engine reads either city.

Collection points
-----------------
21 real locations on the CEG campus and along the Sardar Patel Road /
Gandhi Mandapam Road / Kotturpuram / Saidapet / Velachery corridor, with
coordinates taken from mapping data. Replace or extend the POINTS list with your own surveyed bin sites --
nothing else in the project needs to change.

Waste generation
----------------
Estimated from Greater Chennai Corporation figures rather than copied from
the paper:

    waste_i (m3/day) = population_i * PER_CAPITA_KG / 1000 / MSW_DENSITY

with PER_CAPITA_KG = 0.71 kg/person/day (GCC collects roughly 5,000 t/day
across about 7 million residents) and MSW_DENSITY = 0.40 t/m3 for loose,
uncompacted municipal waste. `population` below is the resident-equivalent
population each collection point serves; adjust these from ward data if your
guide wants a stronger sourcing.

Travel times
------------
`build_time_matrix` prefers the OSRM table service, which uses real Chennai
road geometry. If OSRM is unreachable it falls back to great-circle distance
inflated by a road-detour factor and divided by an average urban speed. The
fallback is clearly reported so it never silently ends up in your results.

Usage:
    python src/instance_builder.py                 # tries OSRM, falls back
    python src/instance_builder.py --offline       # force the fallback
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "chennai_guindy"

# --- waste estimation constants --------------------------------------------
PER_CAPITA_KG = 0.71        # kg/person/day, Greater Chennai Corporation
MSW_DENSITY = 0.40          # t/m3, loose uncompacted municipal solid waste

# --- travel-time fallback constants ----------------------------------------
DETOUR_FACTOR = 1.40        # road distance / straight-line distance, dense urban
AVG_SPEED_KMH = 18.0        # average Chennai arterial speed incl. signals
STOP_PENALTY_MIN = 0.0      # extra per-leg penalty, if you want to model turns

OSRM_URL = "https://router.project-osrm.org/table/v1/driving/"

# --- currency ----------------------------------------------------------------
# The paper's cost parameters (containers.txt weekly bin costs, CCV) are in
# US$. A Chennai build converts them to rupees at the USD->INR mid-market rate
# of the build date, looked up here unless given with --fx. This is an FX
# conversion of Gonzalez et al.'s figures, NOT independently sourced Indian
# rates; meta.json records the rate, its date and source.
DEFAULT_CURRENCY = "INR"
FX_URL = "https://open.er-api.com/v6/latest/USD"

# ============================================================================
# Locally sourced Chennai cost parameters. Leave as None to use the FX
# conversion of Gonzalez et al.'s figures. Replace with Greater Chennai
# Corporation tender rates when available — note that doing so changes the
# bin-vs-routing tradeoff, not just the reported number.
LOCAL_CCV_INR_PER_MIN = None     # compactor + crew operating cost, INR/min
LOCAL_BIN_COST_INR = None        # list of 8 weekly bin costs, INR
# ============================================================================
USER_AGENT = "ceg-guindy-waste-pvrp/1.0 (final-year project)"

# ---------------------------------------------------------------------------
# The depot: the Perungudi dumpsite, where Greater Chennai Corporation
# disposes of municipal solid waste from its southern zones. Every route
# starts and ends here, and the unloading time TU is spent here.
# ---------------------------------------------------------------------------
DEPOT = {"name": "Perungudi MSW disposal site, Greater Chennai Corporation",
         "lat": 12.955663, "lon": 80.226920, "population": 0}

# ---------------------------------------------------------------------------
# Collection points. Coordinates are real; `population` is the resident-
# equivalent catchment each point serves and is the number to refine.
# ---------------------------------------------------------------------------
POINTS = [
    {"name": "College of Engineering Guindy (main gate)", "lat": 13.010940, "lon": 80.235446, "population": 620},
    {"name": "Gandhi Mandapam",                           "lat": 13.006625, "lon": 80.237340, "population": 300},
    {"name": "B.M. Birla Planetarium",                    "lat": 13.011933, "lon": 80.244037, "population": 420},
    {"name": "Kotturpuram (Anna Univ. staff quarters)",   "lat": 13.011458, "lon": 80.243103, "population": 520},
    {"name": "Kotturpuram Urban Forest",                  "lat": 13.023982, "lon": 80.244411, "population": 340},
    {"name": "Ashok Leyland, Sardar Patel Road",          "lat": 13.012486, "lon": 80.222041, "population": 460},
    {"name": "Alexander Square, Little Mount",            "lat": 13.011711, "lon": 80.221195, "population": 390},
    {"name": "Prestige Cosmopolitan, Guindy",             "lat": 13.012038, "lon": 80.222764, "population": 500},
    {"name": "SPIC Building, Anna Salai",                 "lat": 13.011642, "lon": 80.218609, "population": 440},
    {"name": "Kalaignar Arch, Saidapet",                  "lat": 13.020099, "lon": 80.224521, "population": 660},
    {"name": "Karaneeswarar Temple, Saidapet",            "lat": 13.024760, "lon": 80.223176, "population": 560},
    {"name": "ISKCON Saidapet, Jeenis Road",              "lat": 13.020687, "lon": 80.221748, "population": 360},
    {"name": "Kamakshi Amman Temple, Saidapet",           "lat": 13.020016, "lon": 80.220156, "population": 390},
    {"name": "Tholkappia Poonga, RA Puram",               "lat": 13.019192, "lon": 80.264689, "population": 320},
    {"name": "Chemparuthi Hostel, CEG campus",            "lat": 13.013042, "lon": 80.234379, "population": 520},
    {"name": "CEG 5th Block Hostel",                      "lat": 13.014126, "lon": 80.238815, "population": 580},
    {"name": "CEG Main Canteen, CEG Square",              "lat": 13.010464, "lon": 80.236761, "population": 640},
    {"name": "Phoenix Marketcity, Velachery",             "lat": 12.991690, "lon": 80.216942, "population": 700},
    {"name": "Sri Dhandeeswaram Temple, Velachery",       "lat": 12.985750, "lon": 80.223785, "population": 540},
    {"name": "Grand Square Mall, Velachery",              "lat": 12.971821, "lon": 80.220617, "population": 600},
    {"name": "Velachery MRTS Station",                    "lat": 12.967336, "lon": 80.219345, "population": 690},
]


# ---------------------------------------------------------------------------
def estimate_waste(population: int) -> float:
    """Convert a served population into m3/day of municipal solid waste."""
    tonnes = population * PER_CAPITA_KG / 1000.0
    return round(tonnes / MSW_DENSITY, 2)


def check_bin_feasibility(points, max_bin_cap: float, n_rest_days: int = 1) -> list[str]:
    """Warn about points that no single bin combination can serve.

    Because Sunday is a drivers' rest day, even a point collected on every
    working day still holds TWO days of waste on Monday morning (Sunday's
    generation plus Monday's). So the binding constraint is

        (1 + n_rest_days) * W_i  <=  max bin capacity

    With the paper's catalogue (largest combination 5.6 m3) and one rest day,
    that caps a collection point at 2.8 m3/day. Points above that need either
    a smaller catchment, a larger bin, or a second rest-day collection.
    """
    limit = max_bin_cap / (1 + n_rest_days)
    problems = []
    for p in points:
        w = estimate_waste(p["population"])
        if w > limit + 1e-9:
            problems.append(
                f'{p["name"]}: {w:.2f} m3/day exceeds the {limit:.2f} m3/day '
                f'ceiling implied by a {max_bin_cap:.1f} m3 bin and '
                f'{n_rest_days} rest day(s)')
    return problems


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def fallback_matrix(coords: list[tuple[float, float]]) -> np.ndarray:
    """Great-circle distance x detour factor / average speed, in minutes."""
    n = len(coords)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            km = haversine_km(*coords[i], *coords[j]) * DETOUR_FACTOR
            M[i, j] = round(km / AVG_SPEED_KMH * 60.0 + STOP_PENALTY_MIN, 2)
    return M


def get_json(url: str, timeout: int = 60, retries: int = 5,
             backoff: float = 2.0) -> dict:
    """GET a JSON endpoint (OSRM, FX rates) and return it parsed, retrying with
    exponential backoff (2, 4, 8, 16 s ...) when the public demo server
    rate-limits (HTTP 429), has a transient 5xx, or the network drops.
    Client errors other than 429 are not retried -- they will not go away."""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if (exc.code != 429 and exc.code < 500) or attempt == retries:
                raise
            reason = f"HTTP {exc.code}"
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if attempt == retries:
                raise
            reason = str(exc)
        wait = backoff * 2 ** attempt
        print(f"  ! {url.split('/')[2]} {reason}; retrying in {wait:.0f}s "
              f"({attempt + 1}/{retries})")
        time.sleep(wait)
    raise AssertionError("unreachable")


osrm_get = get_json          # name used by exporter.py


def osrm_matrix(coords: list[tuple[float, float]], timeout: int = 60) -> np.ndarray:
    """Real road travel times from the OSRM table service, in minutes.

    OSRM expects lon,lat order. The public demo server rate-limits heavy use;
    for anything bigger than ~50 points, run your own OSRM container.
    """
    locs = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in coords)
    url = f"{OSRM_URL}{locs}?annotations=duration"
    data = get_json(url, timeout=timeout)
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM returned {data.get('code')}")
    M = np.array(data["durations"], dtype=float) / 60.0      # seconds -> minutes
    np.fill_diagonal(M, 0.0)
    return np.round(M, 2)


def build_time_matrix(coords, offline: bool = False) -> tuple[np.ndarray, str]:
    if not offline:
        try:
            return osrm_matrix(coords), "osrm"
        except (urllib.error.URLError, OSError, RuntimeError, TimeoutError) as exc:
            print(f"  ! OSRM unavailable ({exc}); using the haversine fallback.")
    return fallback_matrix(coords), "haversine_fallback"


def lookup_usd_inr() -> tuple[float, str, str]:
    """Today's USD->INR mid-market rate: (rate, date, source URL)."""
    data = get_json(FX_URL, timeout=30)
    if data.get("result") != "success":
        raise RuntimeError(f"FX lookup failed: {data.get('error-type', data)}")
    when = datetime.strptime(data["time_last_update_utc"],
                             "%a, %d %b %Y %H:%M:%S %z").date().isoformat()
    return round(float(data["rates"]["INR"]), 4), when, FX_URL


def currency_meta(currency: str, fx: float | None, fx_date: str | None,
                  fx_source: str | None) -> dict:
    """The currency block of meta.json, read back by loader.load_instance."""
    if currency == "USD":
        return {"currency": "USD", "currency_symbol": "US$", "cost_scale": 1.0,
                "cost_basis": "Gonzalez et al. (2025) US$ cost parameters"}
    if currency != "INR":
        raise ValueError(f"unsupported currency {currency!r} (USD or INR)")

    if fx is None:
        fx, fx_date, fx_source = lookup_usd_inr()
        print(f"  USD->INR mid-market rate {fx} on {fx_date} ({fx_source})")
    meta = {
        "currency": "INR", "currency_symbol": "₹",
        "fx_rate": fx, "fx_date": fx_date or date.today().isoformat(),
        "fx_source": fx_source or "given on the command line (--fx)",
        # cost_scale converts the paper's US$ figures, and the optimizers'
        # penalty weights, into rupees
        "cost_scale": fx,
        "cost_basis": ("FX conversion of Gonzalez et al. (2025) US$ cost "
                       "parameters at fx_rate; not independently sourced "
                       "Indian rates"),
    }
    if LOCAL_CCV_INR_PER_MIN is not None or LOCAL_BIN_COST_INR is not None:
        meta["cost_basis"] = "locally sourced Chennai cost parameters"
        if LOCAL_CCV_INR_PER_MIN is not None:
            meta["ccv"] = float(LOCAL_CCV_INR_PER_MIN)
        if LOCAL_BIN_COST_INR is not None:
            if len(LOCAL_BIN_COST_INR) != 8:
                raise ValueError("LOCAL_BIN_COST_INR needs 8 weekly bin costs")
            meta["bin_cost"] = [float(v) for v in LOCAL_BIN_COST_INR]
    return meta


# ---------------------------------------------------------------------------
def build(out_dir: Path = OUT_DIR, offline: bool = False,
          currency: str = DEFAULT_CURRENCY, fx: float | None = None,
          fx_date: str | None = None, fx_source: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = check_bin_feasibility(POINTS, max_bin_cap=5.6, n_rest_days=1)
    if problems:
        print("  ! these points cannot be served by any single bin combination:")
        for msg in problems:
            print(f"    - {msg}")
        print("    Fix by splitting the catchment, extending containers.txt with "
              "a larger bin, or collecting on the rest day.")

    nodes = [DEPOT] + POINTS
    coords = [(n["lat"], n["lon"]) for n in nodes]

    # --- waste.txt : id, longitude, latitude, waste (m3/day) --------------
    lines = []
    for idx, n in enumerate(nodes):
        waste = 0.00 if idx == 0 else estimate_waste(n["population"])
        lines.append(f"{idx}\t{n['lon']:.6f}\t{n['lat']:.6f}\t{waste:.2f}")
    (out_dir / "waste.txt").write_text("\n".join(lines) + "\n")

    # --- times.txt --------------------------------------------------------
    print(f"  building {len(nodes)}x{len(nodes)} travel-time matrix...")
    M, source = build_time_matrix(coords, offline=offline)
    (out_dir / "times.txt").write_text(
        "\n".join("\t".join(f"{v:.2f}" for v in row) for row in M) + "\n")

    # --- containers.txt : copied from the paper's dataset -----------------
    src = ROOT / "data" / "12_1" / "containers.txt"
    (out_dir / "containers.txt").write_text(
        src.read_text(encoding="utf-8-sig").replace("\r\n", "\n"))

    # --- points.csv : human-readable reference + UI labels ----------------
    csv_lines = ["id,name,lat,lon,population,waste_m3_day"]
    for idx, n in enumerate(nodes):
        waste = 0.00 if idx == 0 else estimate_waste(n["population"])
        csv_lines.append(
            f'{idx},"{n["name"]}",{n["lat"]:.6f},{n["lon"]:.6f},'
            f'{n["population"]},{waste:.2f}')
    (out_dir / "points.csv").write_text("\n".join(csv_lines) + "\n")

    # --- provenance -------------------------------------------------------
    (out_dir / "meta.json").write_text(json.dumps({
        "instance": "chennai_guindy",
        "n_points": len(POINTS),
        "travel_time_source": source,
        "per_capita_kg_per_day": PER_CAPITA_KG,
        "msw_density_t_per_m3": MSW_DENSITY,
        "detour_factor": DETOUR_FACTOR if source != "osrm" else None,
        "avg_speed_kmh": AVG_SPEED_KMH if source != "osrm" else None,
        "bin_combinations_source": "Gonzalez et al. (2025) dataset, containers.txt",
        **currency_meta(currency, fx, fx_date, fx_source),
        "depot": DEPOT["name"],
        "depot_note": "Perungudi dumpsite, the disposal site for Greater "
                      "Chennai Corporation's southern zones",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    total = sum(estimate_waste(p["population"]) for p in POINTS)
    print(f"  wrote {out_dir}")
    print(f"  {len(POINTS)} collection points, {total:.2f} m3/day total")
    print(f"  travel times from: {source}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="skip OSRM and use the haversine fallback")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--currency", choices=["INR", "USD"], default=DEFAULT_CURRENCY,
                    help="currency of the instance's costs (default INR)")
    ap.add_argument("--fx", type=float, default=None,
                    help="USD->INR rate; looked up for today if omitted")
    ap.add_argument("--fx-date", default=None, help="date of the --fx rate")
    ap.add_argument("--fx-source", default=None, help="source of the --fx rate")
    args = ap.parse_args()
    build(Path(args.out), offline=args.offline, currency=args.currency,
          fx=args.fx, fx_date=args.fx_date, fx_source=args.fx_source)
