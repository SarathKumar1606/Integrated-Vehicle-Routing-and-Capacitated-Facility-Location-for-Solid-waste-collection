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

import math
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
    ids = np.array([int(r[0]) for r in waste_rows])
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

    Q = Q if Q is not None else vehicle_capacity(n_points)
    n_vehicles = n_vehicles if n_vehicles is not None else fleet_size(n_points)
    n_work = N_DAYS - len(REST_DAYS)
    TL = TL if TL is not None else shift_length(C, n_vehicles, n_work)

    return Instance(
        name=name, n_points=n_points, ids=ids, lon=lon, lat=lat, W=W, C=C,
        bin_cap=bin_cap, bin_service=bin_service, bin_cost=bin_cost,
        Q=Q, n_vehicles=n_vehicles, TL=TL,
    )


if __name__ == "__main__":
    inst = load_instance(Path(__file__).parent.parent / "data" / "12_1", "i.12.1")
    print(inst.summary())
