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

## Task 2 — depot and points

- Depot moved to the Perungudi MSW disposal site (12.955663, 80.226920).
- Removed Besant Nagar Beach, Broken Bridge and Adyar Banyan Tree; added
  Chemparuthi Hostel, CEG 5th Block Hostel and CEG Main Canteen in their
  place (21 points).
- `check_bin_feasibility`: no point exceeds the 2.80 m³/day ceiling. Largest
  are Phoenix Marketcity 1.24, Velachery MRTS 1.22 and Kalaignar Arch 1.17
  m³/day; total 18.71 m³/day (was 17.72).
- `data/chennai_guindy/` is rebuilt in task 3, together with the
  peak-hour instance, so OSRM is queried once per instance.

## Task 3 — peak-hour scenario

Both instances built from OSRM (`travel_time_source: osrm`), in rupees at
96.3789 ₹/US$ (2026-10-02, open.er-api.com):

| Instance | traffic_multiplier | Σ travel-time matrix (min) | nV | TL from Eq. (10) | Depot → CEG main gate |
|---|---|---|---|---|---|
| chennai_guindy | 1.0 | 3,436.61 | 3 | **96 min** | 14.13 min |
| chennai_guindy_peak | 1.5 | 5,155.01 | 3 | **144 min** | 21.20 min |

The peak matrix is exactly round(1.5 × free-flow, 2). Eq. (10) sets
TL = ⌈Σ C_ij / (nV (nV − 1) |T − T'|)⌉ = ⌈Σ C_ij / 36⌉, so TL scales with
the matrix: ⌈95.46⌉ = 96 and ⌈143.19⌉ = 144. That is paper-consistent, but
it means the peak scenario's shift limit grows with congestion. For the
report: under Eq. (10), slower traffic does not make the shift limit
harder to meet; it mostly shows up as routing cost (CCV × minutes).

Moving the depot to Perungudi also lengthens every route: the depot is now
14 min from the CEG gate (5 min from the old placeholder), so each route
carries about 25–30 min of extra depot travel, compared with the v1.0 Chennai results.

## Task 4 — experiments

Driver: for each Chennai instance, SA × 30, then GA generations =
round(mean SA evaluations / 100) − 1 from that instance's own 30 SA runs,
then GA × 30 (`--iters-per-temp 2500 --workers 11`; each runner call retried
once if it fails). Status lines are in `results/drive_status.txt`.
The v1.0 (US$, placeholder-depot) Chennai results were moved to
`results/superseded/v1.0_chennai_usd/`; i.12.1 was not re-run.

### Laptop on battery: runs ~5× slower than in v1.0

Each SA run took ~230–275 s against ~43 s in the v1.0 runs, with the same
evaluation counts (~510k). Cause: the laptop was running on battery
(Win32_Battery status "discharging", 92%, Balanced power plan), which
throttles the CPU. The code was ruled out: under the same load a fitness
evaluation measured 24 µs (13 µs idle on mains in v1.0). Expected effect:
the GA phase takes ~1 h per instance instead of ~13 min, and the battery
may not last. A desktop notification asked for the charger to be plugged
in. Runtimes in the summary are therefore not comparable with v1.0's.

### If the run is cut off: how to resume

At 19:15 the battery was at 79% and falling ~13% per 10 min, so the run is
unlikely to finish on battery. `runner.py` now has `--resume`: seeds that
already have a CSV row and a saved solution are skipped, and new rows are
appended. Every finished seed is on disk, so after plugging in, re-run any
step that did not reach "ALL DONE" in `results/drive_status.txt`:

    python src/runner.py --instance data/chennai_guindy --algo ga --runs 30 --iters-per-temp 2500 --generations 5219 --workers 11 --resume
    python src/runner.py --instance data/chennai_guindy_peak --algo sa --runs 30 --iters-per-temp 2500 --workers 11 --resume
    # then calibrate GA from results/chennai_guindy_peak_sa.csv:
    #   generations = round(mean(evaluations) / 100) - 1
    python src/runner.py --instance data/chennai_guindy_peak --algo ga --runs 30 --iters-per-temp 2500 --generations <G> --workers 11 --resume

### Outcome

The battery lasted: all runs finished at 20:05, exit code 0, no retries
needed, every run feasible. GA generations were recalibrated from each
instance's own 30-run SA mean:

| Instance | SA mean evals (range) | GA `--generations` | GA evals | GA vs SA |
|---|---|---|---|---|
| chennai_guindy | 522,000 (490,000–550,000) | 5219 | 522,000 | +0.00% |
| chennai_guindy_peak | 517,333 (467,500–555,000) | 5172 | 517,300 | −0.01% |

The driver is saved as `scripts/run_chennai_experiments.sh`. Runtimes were
measured on battery power, so they are 3–6× those of the v1.0 runs.

## Task 5 — summary and maps

`results/summary.md` regenerated with i.12.1 (US$, unchanged) and both
Chennai scenarios (₹). Map JSONs re-exported from the best solutions:
`ui/data/chennai_guindy.json` (SA seed 16, ₹29,250.84) and
`ui/data/chennai_guindy_peak.json` (SA seed 20, ₹37,378.69), with OSRM
street geometry on all routes.

Note for the map: each route's `osrm_duration_min` is OSRM's own free-flow
estimate for that street path. The arrive/depart times and `duration_min`
use the instance's matrix, i.e. ×1.5 in the peak scenario.

`ui/index.html` was not changed. It embeds the v1.0 plan (US$, placeholder
depot) and labels costs "US$ per week", so it does not show these
results until it is regenerated from the new JSONs.

Appendix A validation at the end of the session: 7/7. Decoder equivalence: BIT-FOR-BIT IDENTICAL on i.12.1, chennai_guindy and chennai_guindy_peak.

---

# 2026-10-03/04 — peak traffic with the shift limit held at 96 min

Scenario `chennai_guindy_peak_tl96`: the peak instance loaded with
`load_instance(..., TL=96)` (`runner.py --tl 96`), SA × 30 then GA × 30,
GA generations from its own SA mean: 531,917 evals → `--generations 5318`
(531,900, −0.00%). Run with `scripts/run_scenario.sh`. 22:55 → 00:07.
The laptop started on battery and was plugged in at 23:19, but the CPU ran
at ~40–55% of its rated speed throughout, so GA runs took ~22 min each.

Results: SA 30/30 feasible; GA 29/30 feasible. GA seed 9 ended with one
route of 96.37 min (0.37 min over), penalty ₹35,660, fitness ₹90,081.

The limit hardly binds for good solutions. Only 2 of 30 plain-peak SA
solutions would break 96 min, and SA cost plain vs pinned is not
significantly different (p = 0.70). It does bind for GA: all 30 plain-peak
GA solutions would break it (longest 126 min). Pinned, GA uses 8–10 routes
instead of 7, and its median cost is +7.7% (p = 1.7e-9).

Fixed along the way: summarize.py and exporter.py found an instance's
solutions with a prefix glob (`{name}_*`), which also matched
`chennai_guindy_peak_*` when looking for `chennai_guindy`, and would have
matched `chennai_guindy_peak_tl96_*` for `chennai_guindy_peak`. Both now
match `{name}_{sa,ga}_seed*.npz` exactly. The v2.0 summary and maps were not
affected: re-checked, they pick the same best solutions (seeds 16 and 20).
