"""
M5 - Optimizers: Simulated Annealing and Genetic Algorithm
===========================================================
Both use the paper's mixed encoding -- a permutation chromosome `pop` and a
binary chromosome `mask` -- and both call `decoder.decode()` to score a
candidate. Neither knows anything about waste; they just shuffle matrices.

SA settings from the paper's tuning (Section 5.3):
    solution update = exchange mutation (EM), cooling factor alpha = 0.9,
    acceptance rate chi = 0.8, initial temperature from Eq. (8)

GA settings from the paper's tuning (Section 5.2):
    crossover = cycle crossover (CX) at rate 0.8, mutation = exchange (EM)
    at rate 0.05, tournament selection (size 2), population 100, elitism 2
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from decoder import accumulate, decode
from loader import Instance


# ---------------------------------------------------------------------------
# Chromosome helpers
# ---------------------------------------------------------------------------
def random_solution(inst: Instance, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    nI, nD = inst.n_points, inst.n_work_days
    pop = np.column_stack([rng.permutation(nI) for _ in range(nD)])
    mask = (rng.random((nI, nD)) < 0.5).astype(np.int8)
    return pop, repair(inst, mask)


def repair(inst: Instance, mask: np.ndarray) -> np.ndarray:
    """Add visits until no collection point overflows the largest available bin.

    This is the paper's repair step (Section 4.2): infeasibility with respect
    to Eqs. (2c) and (2i)-(2m) is fixed by adding ones to the mask, rather
    than by penalising it. A point that is never visited would accumulate
    waste forever, so this also guarantees every point gets at least one visit.
    """
    mask = mask.copy()
    max_cap = inst.bin_cap.max()
    work_days = list(inst.work_days)

    for _ in range(inst.n_work_days + 1):
        w = accumulate(inst, mask)
        wmax = w.max(axis=1)
        bad = np.where(wmax > max_cap + 1e-9)[0]
        if bad.size == 0:
            break
        for i in bad:
            if mask[i].all():
                continue                                    # nothing left to add
            # add a visit on the working day where the pile is currently worst
            w_work = np.array([w[i, d] for d in work_days])
            w_work[mask[i] == 1] = -np.inf                  # only free days
            mask[i, int(np.argmax(w_work))] = 1
    return mask


def fitness(inst: Instance, pop: np.ndarray, mask: np.ndarray,
            lam: float = 100.0, gamma: float = 1000.0) -> float:
    return decode(inst, pop, mask, lam, gamma).fitness


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    algorithm: str
    best_fitness: float
    best_pop: np.ndarray
    best_mask: np.ndarray
    evaluations: int
    history: list[float]
    seed: int
    T0: float | None = None


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------
def _neighbour(inst: Instance, pop: np.ndarray, mask: np.ndarray,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One move: either an exchange mutation on a day's visiting order, or a
    uniform flip of one collect/don't-collect bit."""
    pop, mask = pop.copy(), mask.copy()
    d = rng.integers(inst.n_work_days)
    if rng.random() < 0.5:
        a, b = rng.choice(inst.n_points, size=2, replace=False)
        pop[a, d], pop[b, d] = pop[b, d], pop[a, d]          # exchange mutation
    else:
        i = rng.integers(inst.n_points)
        mask[i, d] ^= 1
        mask = repair(inst, mask)
    return pop, mask


def initial_temperature(inst: Instance, rng: np.random.Generator,
                        m: int = 100, chi: float = 0.8) -> float:
    """Eq. (8): approximate T0 from the average uphill move, assuming cooling
    in polynomial time (Delahaye et al., 2019)."""
    pop, mask = random_solution(inst, rng)
    f = fitness(inst, pop, mask)
    uphill, m1 = [], 0
    for _ in range(m):
        np_, nm = _neighbour(inst, pop, mask, rng)
        fn = fitness(inst, np_, nm)
        if fn < f:
            m1 += 1
        else:
            uphill.append(fn - f)
        pop, mask, f = np_, nm, fn
    if not uphill:
        return 100.0
    df = float(np.mean(uphill))
    denom = (m - m1) * chi - m1 * (1 - chi)
    if denom <= 0:
        return max(df, 1.0) * 10
    ratio = (m - m1) / denom
    return df / math.log(ratio) if ratio > 1 else max(df, 1.0) * 10


def simulated_annealing(inst: Instance, seed: int = 0, alpha: float = 0.9,
                        iters_per_temp: int = 200, t_final: float = 1e-6,
                        max_stall: int = 100,
                        lam: float = 100.0, gamma: float = 1000.0) -> RunResult:
    rng = np.random.default_rng(seed)
    T0 = initial_temperature(inst, rng)
    T = T0

    pop, mask = random_solution(inst, rng)
    f = fitness(inst, pop, mask, lam, gamma)
    best_f, best_pop, best_mask = f, pop.copy(), mask.copy()

    evals, stall, history = 0, 0, [best_f]
    while T > t_final:
        improved = False
        for _ in range(iters_per_temp):
            np_, nm = _neighbour(inst, pop, mask, rng)
            fn = fitness(inst, np_, nm, lam, gamma)
            evals += 1
            delta = fn - f
            if delta < 0 or rng.random() < math.exp(-delta / T):   # Metropolis
                pop, mask, f = np_, nm, fn
                if f < best_f:
                    best_f, best_pop, best_mask = f, pop.copy(), mask.copy()
                    improved = True
        history.append(best_f)
        T *= alpha
        stall = 0 if improved else stall + 1
        if stall >= max_stall:
            break

    return RunResult("SA", best_f, best_pop, best_mask, evals, history, seed, T0)


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------
def cycle_crossover(p1: np.ndarray, p2: np.ndarray,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Cycle Crossover (CX) -- the paper's best-performing operator."""
    n = len(p1)
    c1, c2 = p1.copy(), p2.copy()
    pos = {v: i for i, v in enumerate(p1)}
    visited = np.zeros(n, dtype=bool)
    cycle = 0
    for start in range(n):
        if visited[start]:
            continue
        idx, cyc = start, []
        while not visited[idx]:
            visited[idx] = True
            cyc.append(idx)
            idx = pos[p2[idx]]
        if cycle % 2 == 1:                       # alternate cycles are swapped
            for j in cyc:
                c1[j], c2[j] = p2[j], p1[j]
        cycle += 1
    return c1, c2


def two_point_crossover(a: np.ndarray, b: np.ndarray,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(a)
    if n < 2:
        return a.copy(), b.copy()
    i, j = sorted(rng.choice(n, size=2, replace=False))
    c1, c2 = a.copy(), b.copy()
    c1[i:j], c2[i:j] = b[i:j], a[i:j]
    return c1, c2


def _crossover(inst, P1, M1, P2, M2, rng, rate):
    if rng.random() > rate:
        return (P1.copy(), M1.copy()), (P2.copy(), M2.copy())
    C1p, C2p = P1.copy(), P2.copy()
    C1m, C2m = M1.copy(), M2.copy()
    for d in range(inst.n_work_days):
        C1p[:, d], C2p[:, d] = cycle_crossover(P1[:, d], P2[:, d], rng)
        C1m[:, d], C2m[:, d] = two_point_crossover(M1[:, d], M2[:, d], rng)
    return (C1p, C1m), (C2p, C2m)


def _mutate(inst, P, M, rng, perm_rate, bit_rate):
    for d in range(inst.n_work_days):
        if rng.random() < perm_rate:                    # exchange mutation
            a, b = rng.choice(inst.n_points, size=2, replace=False)
            P[a, d], P[b, d] = P[b, d], P[a, d]
        flips = rng.random(inst.n_points) < bit_rate    # uniform mutation
        M[flips, d] ^= 1
    return P, repair(inst, M)


def genetic_algorithm(inst: Instance, seed: int = 0, pop_size: int = 100,
                      generations: int = 300, crossover_rate: float = 0.8,
                      mutation_rate: float = 0.05, elite: int = 2,
                      lam: float = 100.0, gamma: float = 1000.0) -> RunResult:
    rng = np.random.default_rng(seed)
    bit_rate = 1.0 / inst.n_points

    population = [random_solution(inst, rng) for _ in range(pop_size)]
    scores = np.array([fitness(inst, p, m, lam, gamma) for p, m in population])
    evals = pop_size

    order = np.argsort(scores)
    best_f = float(scores[order[0]])
    best_pop, best_mask = population[order[0]]
    best_pop, best_mask = best_pop.copy(), best_mask.copy()
    history = [best_f]

    for _ in range(generations):
        # --- elitism ------------------------------------------------------
        order = np.argsort(scores)
        new_pop = [(population[i][0].copy(), population[i][1].copy())
                   for i in order[:elite]]

        # --- tournament selection, crossover, mutation --------------------
        while len(new_pop) < pop_size:
            i1, i2 = rng.choice(pop_size, size=2, replace=False)
            a = i1 if scores[i1] <= scores[i2] else i2
            j1, j2 = rng.choice(pop_size, size=2, replace=False)
            b = j1 if scores[j1] <= scores[j2] else j2

            (P1, M1), (P2, M2) = _crossover(
                inst, *population[a], *population[b], rng, crossover_rate)
            new_pop.append(_mutate(inst, P1, M1, rng, mutation_rate, bit_rate))
            if len(new_pop) < pop_size:
                new_pop.append(_mutate(inst, P2, M2, rng, mutation_rate, bit_rate))

        population = new_pop
        scores = np.array([fitness(inst, p, m, lam, gamma) for p, m in population])
        evals += pop_size

        k = int(np.argmin(scores))
        if scores[k] < best_f:
            best_f = float(scores[k])
            best_pop, best_mask = population[k][0].copy(), population[k][1].copy()
        history.append(best_f)

    return RunResult("GA", best_f, best_pop, best_mask, evals, history, seed)
