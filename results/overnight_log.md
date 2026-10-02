# Overnight log

## Task 2 — experiments

### i.12.1 GA run crashed (BrokenProcessPool)

The first launch of

    python src/runner.py --instance data/12_1 --name i.12.1 --algo both --runs 30 --iters-per-temp 2500 --generations 3399 --workers 11

completed all 30 SA runs, then the GA half died immediately:

    concurrent.futures.process.BrokenProcessPool: A process in the process
    pool was terminated abruptly while the future was running or pending.

No GA row was written (`results/i.12.1_ga.csv` empty, no `i.12.1_ga_*.npz`).
The Chennai command (SA + GA) started afterwards and ran normally.
Action: re-run the i.12.1 GA half on its own after Chennai finishes.

### Wall-time estimate was too optimistic

The estimate (≈7–10 min total) used single-run timings. With 11 workers in
parallel each run took about 4× longer (i.12.1 SA: 41.6 s mean per run vs
10.3 s alone), which is contention for shared cores, cache and memory
bandwidth on a 12-core / 16-thread CPU.

### Single-run calibration missed the 5% evaluation-matching target

The calibration (`results/calibration_*.txt`) measured SA on one seed (seed 0)
and set GA's generations from it. SA stops on its own schedule (temperature
floor or 100 non-improving temperature steps), so its evaluation count
varies by seed, and seed 0 was on the low side:

| Instance | SA evals, seed 0 | SA evals, mean of 30 | GA evals | GA vs SA |
|---|---|---|---|---|
| i.12.1 | 340,000 | 445,583 | 340,000 | −23.7% |
| chennai_guindy | 467,500 | 522,833 (460,000–555,000) | 467,500 | −10.6% |

Both miss the ±5% requirement. GA was recalibrated to the 30-run SA mean,
i.e. generations = round(mean SA evals / 100) − 1:

- i.12.1: `--generations 4455` → 445,600 evaluations
- chennai_guindy: `--generations 5227` → 522,800 evaluations

The i.12.1 GA re-run at 3399 generations was stopped part-way. The
uncalibrated Chennai GA results (and the partial i.12.1 GA files) were moved
to `results/superseded/` rather than deleted. SA results are unchanged.

## Outcome

Nothing is left failing. All four tasks completed and were committed:
the loader fix (bit-for-bit identical on all 12 dataset instances), 4 × 30
feasible runs matched on evaluations, `results/summary.md` with figures,
and the map JSON re-exported from the best solution (SA seed 28,
223.72 US$/week, OSRM geometry on all 7 routes).

---

# Session 2026-10-02 — rupees, real depot, peak-hour scenario

Appendix A validation before starting: 7/7.

## Task 1 — currency

- USD→INR rate: **96.3789**, mid-market, 2026-10-02, from
  https://open.er-api.com/v6/latest/USD. Cross-check: the ECB reference
  rate via api.frankfurter.dev was 96.33 for 2026-10-01 (−0.05%; today's ECB
  fix was not yet published).
- **Decision: penalty weights and SA's final temperature are scaled with
  the costs.** Converting `bin_cost` and `ccv` to rupees multiplies every
  cost by ~96, but the penalty weights of Eq. (7) (λ = 100, γ = 1000) and
  SA's stopping temperature (`t_final = 1e-6`) are in the same money units.
  Left unscaled, a fleet-size violation would cost ₹33 against routes
  costing thousands of rupees, and the optimizer would be solving a
  different, much less constrained problem. `optimizers.penalty_weights()`
  and SA's stopping test therefore multiply them by `inst.cost_scale`. This
  is in optimizers.py, runner.py and exporter.py; the decoder's model is
  untouched. For the paper's instances `cost_scale` = 1.0 exactly, and seeded
  SA/GA runs on i.12.1 were checked to be bit-identical to v1.0.
- Scale invariance check (old Chennai instance, scratch copy): the best
  solution decodes to the same bins and routes, at exactly 96.3789× the
  cost (223.72 US$ → ₹21,561.85). A seeded SA run spends the same number of
  evaluations but does not follow an identical trajectory: rounding in the
  scaled values flips an occasional Metropolis decision, and SA amplifies
  that. The rupee experiments are the same problem, not a replay of the
  US$ runs.
