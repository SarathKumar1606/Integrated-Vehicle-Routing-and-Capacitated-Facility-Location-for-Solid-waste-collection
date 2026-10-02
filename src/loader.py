"""
M3 - Instance Loader
====================
Reads an instance folder (waste.txt, times.txt, containers.txt) into an
`Instance` object, and attaches the fixed model parameters from the paper.

File formats (per the dataset README):
  waste.txt      : id, longitude, latitude, daily waste generation (m3/day)
                   -- row 0 is the DEPOT (waste = 0.00)
  times.txt      : (nI+1) x (nI+1) from/to matrix of travel times in minutes
                   -- index 0 is the depot
  containers.txt : id, capacity (m3), service time (min), weekly cost (US$)

The same loader reads Bahia Blanca and Chennai instances -- that is the whole
point: the engine never knows which city it is looking at.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Fixed model parameters (Gonzalez et al., 2025, Section 5.1)
# ---------------------------------------------------------------------------
CCV = 0.5764        # collection vehicle cost, US$ per minute
TU = 8.0            # unloading time at the depot, minutes
N_DAYS = 7          # planning horizon: one week
REST_DAYS = (6,)    # 0=Mon ... 6=Sun -> Sunday is the drivers' rest day
DAY_NAMES = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")

CURRENCY_SYMBOLS = {"USD": "US$", "INR": "₹"}

# Windows consoles default to a legacy code page that cannot print "₹";
# every script imports this module, so switch stdout/stderr to UTF-8 here.
for _stream in (sys.stdout, sys.stderr):
    try:
        if (_stream.encoding or "").lower().replace("-", "") != "utf8":
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def indian_grouping(x: float, decimals: int = 2) -> str:
    """1234567.891 -> '12,34,567.89' (lakh/crore grouping)."""
    s = f"{abs(x):.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return ("-" if x < 0 else "") + whole + ("." + frac if frac else "")


def format_money(x: float, currency: str = "USD", symbol: str | None = None,
                 decimals: int = 2) -> str:
    """Rupees as '₹1,23,456.78'; anything else as '123456.78 US$'."""
    symbol = symbol or CURRENCY_SYMBOLS.get(currency, currency)
    if currency == "INR":
        return f"{'-' if x < 0 else ''}{symbol}{indian_grouping(abs(x), decimals)}"
    return f"{x:.{decimals}f} {symbol}"


@dataclass
class Instance:
    """Everything the optimizer needs to know about one problem instance."""

    name: str
    n_points: int                  # nI, excluding the depot
    ids: np.ndarray                # original ids from waste.txt (for reference)
    lon: np.ndarray                # longitude, index 0 = depot
    lat: np.ndarray                # latitude,  index 0 = depot
    W: np.ndarray                  # daily waste generation, shape (nI+1,), W[0] = 0
    C: np.ndarray                  # travel-time matrix, shape (nI+1, nI+1)

    bin_cap: np.ndarray            # CAP_b, capacity of each bin combination
    bin_service: np.ndarray        # S_b, service time of each bin combination
    bin_cost: np.ndarray           # CIN_b, weekly cost of each bin combination

    Q: float                       # vehicle capacity, m3
    n_vehicles: int                # nV, fleet size
    TL: float                      # shift length, minutes

    ccv: float = CCV
    tu: float = TU
    n_days: int = N_DAYS
    rest_days: tuple = REST_DAYS

    # ---- currency --------------------------------------------------------
    # Costs (bin_cost, ccv) are in `currency`. cost_scale is the factor that
    # converted the paper's US$ figures into it (1.0 for the paper's own
    # instances); optimizers use it to express the penalty weights and SA's
    # final temperature in the same currency, so a rescaled instance is the
    # same optimisation problem.
    currency: str = "USD"
    currency_symbol: str = "US$"
    cost_scale: float = 1.0
    fx_rate: float | None = None

    def money(self, x: float, decimals: int = 2) -> str:
        return format_money(x, self.currency, self.currency_symbol, decimals)

    # ---- derived ---------------------------------------------------------
    work_days: tuple = field(init=False)

    def __post_init__(self) -> None:
        self.work_days = tuple(d for d in range(self.n_days)
                               if d not in self.rest_days)

    @property
    def n_work_days(self) -> int:
        return len(self.work_days)

    @property
    def n_bins(self) -> int:
        return len(self.bin_cap)

    def summary(self) -> str:
        return (
            f"Instance {self.name}\n"
            f"  collection points : {self.n_points}\n"
            f"  bin combinations  : {self.n_bins}\n"
            f"  vehicle capacity  : {self.Q:.1f} m3\n"
            f"  fleet size (nV)   : {self.n_vehicles}\n"
            f"  shift length (TL) : {self.TL:.2f} min\n"
            f"  working days      : {[DAY_NAMES[d] for d in self.work_days]}\n"
            f"  total daily waste : {self.W.sum():.2f} m3/day"
        )


# ---------------------------------------------------------------------------
# Fleet size and shift length -- Eqs. (9) and (10) of the paper
# ---------------------------------------------------------------------------
def fleet_size(n_points: int) -> int:
    """Eq. (9):  nV = ceil(nI / 10)."""
    return math.ceil(n_points / 10)


def shift_length(C: np.ndarray, n_vehicles: int, n_work_days: int) -> float:
    """Eq. (10): TL = ceil( sum(C_ij) / (nV * (nV - 1) * |T - T'|) ).

    Note: for nV == 1 the paper's denominator collapses to zero. We fall back
    to nV * nV in that case, which keeps the formula well defined and matches
    the intent (a single vehicle still needs a workable shift).
    """
    denom = n_vehicles * (n_vehicles - 1) * n_work_days
    if denom <= 0:
        denom = n_vehicles * n_vehicles * n_work_days
    return math.ceil(C.sum() / denom)


def vehicle_capacity(n_points: int) -> float:
    """Section 5.1: 12 m3 for the 12-point instances, 15 m3 for the 15-point
    instances, 21 m3 for everything larger."""
    if n_points <= 12:
        return 12.0
    if n_points <= 15:
        return 15.0
    return 21.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read_rows(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    return [ln.split() for ln in text.split("\n") if ln.strip()]


def load_instance(folder: str | Path,
                  name: str | None = None,
                  Q: float | None = None,
                  n_vehicles: int | None = None,
                  TL: float | None = None) -> Instance:
    """Load an instance folder. Q / n_vehicles / TL may be overridden, which is
    useful for the Chennai instance where local fleet data may be known."""
    folder = Path(folder)
    name = name or folder.name

    waste_rows = _read_rows(folder / "waste.txt")
    # The larger instances (40, 80, 120, 163 points) label the depot row
    # "Depot" rather than 0; keep such a label as a string. Every other row
    # must have an integer id.
    depot_id = waste_rows[0][0]
    try:
        depot_id = int(depot_id)
    except ValueError:
        pass
    point_ids = [int(r[0]) for r in waste_rows[1:]]
    ids = np.array([depot_id] + point_ids,
                   dtype=object if isinstance(depot_id, str) else None)
    lon = np.array([float(r[1]) for r in waste_rows])
    lat = np.array([float(r[2]) for r in waste_rows])
    W = np.array([float(r[3]) for r in waste_rows])

    n_points = len(waste_rows) - 1          # row 0 is the depot
    if W[0] != 0.0:
        raise ValueError("Row 0 of waste.txt must be the depot with waste 0.00")

    C = np.array([[float(v) for v in r] for r in _read_rows(folder / "times.txt")])
    if C.shape != (n_points + 1, n_points + 1):
        raise ValueError(
            f"times.txt is {C.shape}, expected {(n_points + 1, n_points + 1)}")

    bin_rows = _read_rows(folder / "containers.txt")
    bin_cap = np.array([float(r[1]) for r in bin_rows])
    bin_service = np.array([float(r[2]) for r in bin_rows])
    bin_cost = np.array([float(r[3]) for r in bin_rows])

    # --- optional currency conversion from meta.json ----------------------
    # containers.txt and CCV are in the paper's US$. An instance may declare
    # another currency: cost_scale multiplies both at load time, and explicit
    # `ccv` / `bin_cost` entries (locally sourced figures) override them.
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    currency = meta.get("currency") or "USD"
    cost_scale = float(meta.get("cost_scale") or 1.0)
    ccv = CCV * cost_scale
    bin_cost = bin_cost * cost_scale
    if meta.get("ccv") is not None:
        ccv = float(meta["ccv"])
    if meta.get("bin_cost") is not None:
        bin_cost = np.array([float(v) for v in meta["bin_cost"]])
        if bin_cost.shape != bin_cap.shape:
            raise ValueError("meta.json bin_cost must list one cost per bin combination")

    Q = Q if Q is not None else vehicle_capacity(n_points)
    n_vehicles = n_vehicles if n_vehicles is not None else fleet_size(n_points)
    n_work = N_DAYS - len(REST_DAYS)
    TL = TL if TL is not None else shift_length(C, n_vehicles, n_work)

    return Instance(
        name=name, n_points=n_points, ids=ids, lon=lon, lat=lat, W=W, C=C,
        bin_cap=bin_cap, bin_service=bin_service, bin_cost=bin_cost,
        Q=Q, n_vehicles=n_vehicles, TL=TL, ccv=ccv,
        currency=currency,
        currency_symbol=meta.get("currency_symbol")
                        or CURRENCY_SYMBOLS.get(currency, currency),
        cost_scale=cost_scale,
        fx_rate=meta.get("fx_rate"),
    )


if __name__ == "__main__":
    inst = load_instance(Path(__file__).parent.parent / "data" / "12_1", "i.12.1")
    print(inst.summary())
