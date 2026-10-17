"""
Summarize the experiments into results/summary.md and convergence plots.

Reads results/<name>_<algo>.csv (one row per seed, written by runner.py) and
results/solutions/<name>_<algo>_seed<N>.npz (best chromosome + history).

Usage:
    python src/summarize.py
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from scipy.stats import mannwhitneyu     # noqa: E402

from decoder import Solution, decode                         # noqa: E402
from loader import Instance, indian_grouping, load_instance  # noqa: E402
from optimizers import penalty_weights                       # noqa: E402
from runner import summarize             # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
# runner.py name -> instance folder
INSTANCES = {"i.12.1": ROOT / "data" / "12_1",
             "chennai_guindy": ROOT / "data" / "chennai_guindy",
             "chennai_guindy_peak": ROOT / "data" / "chennai_guindy_peak",
             "chennai_guindy_peak_tl96": ROOT / "data" / "chennai_guindy_peak"}
# load_instance overrides a runner.py name was run with (runner.py --tl)
OVERRIDES = {"chennai_guindy_peak_tl96": {"TL": 96.0}}
SCENARIOS = [("Free-flow", "chennai_guindy"),
             ("Peak (×1.5)", "chennai_guindy_peak"),
             ("Peak, shift limit held at 96 min", "chennai_guindy_peak_tl96")]
ALGOS = ["sa", "ga"]

# Gonzalez et al. (2025): Table 9 (SA on i.12.1) and Table 10 (MILP optimum)
PAPER_SA_MIN, PAPER_SA_MEAN, PAPER_MILP = 188.0, 193.7, 178.42

# Palette: categorical slots 1-2 of the validated default palette, fixed order
COLOR = {"sa": "#2a78d6", "ga": "#eb6834"}
LABEL = {"sa": "SA", "ga": "GA"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def read_rows(name: str, algo: str) -> list[dict]:
    path = RESULTS / f"{name}_{algo}.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("overall_cost", "bin_cost", "routing_cost", "runtime_s"):
            r[k] = float(r[k])
        r["seed"] = int(r["seed"])
        r["evaluations"] = int(r["evaluations"])
        r["feasible"] = r["feasible"] == "True"
        # columns added for the TL experiment; absent in older CSVs
        for k in ("penalty", "shift_limit_min", "longest_route_min"):
            if k in r:
                r[k] = float(r[k])
        for k in ("n_routes", "routes_over_tl"):
            if k in r:
                r[k] = int(r[k])
    return sorted(rows, key=lambda r: r["seed"])


_INST: dict[str, Instance] = {}


def inst_of(name: str) -> Instance:
    """The instance behind a runner.py name -- for its currency and TL."""
    if name not in _INST:
        _INST[name] = load_instance(INSTANCES[name], name,
                                    **OVERRIDES.get(name, {}))
    return _INST[name]


def meta_of(name: str) -> dict:
    path = INSTANCES[name] / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def json_traffic(name: str) -> str:
    return f"{meta_of(name).get('traffic_multiplier', 1.0)}"


def money(name: str, x: float) -> str:
    return inst_of(name).money(x)


def best_solution(name: str) -> tuple[str, Solution] | None:
    """Best saved solution for an instance (feasible first, then cheapest),
    the same rule exporter.py uses."""
    inst, best = inst_of(name), None
    for f in solution_files(name):
        with np.load(f) as z:
            sol = decode(inst, z["pop"], z["mask"], *penalty_weights(inst))
        key = (not sol.feasible, sol.fitness)
        if best is None or key < best[0]:
            best = (key, f.stem, sol)
    return None if best is None else (best[1], best[2])


def solution_files(name: str) -> list[Path]:
    """Saved runs of exactly this instance. A prefix glob such as
    f"{name}_*" would also catch chennai_guindy_peak_* for chennai_guindy."""
    return sorted(f for algo in ALGOS
                  for f in (RESULTS / "solutions").glob(f"{name}_{algo}_seed*.npz"))


def run_stats(name: str) -> dict:
    """Decode every saved run of an instance (SA and GA): how many are
    feasible, and how many routes run past the shift limit."""
    inst, st = inst_of(name), {"runs": 0, "feasible": 0, "runs_over": 0,
                               "routes_over": 0, "routes": 0}
    for f in solution_files(name):
        with np.load(f) as z:
            sol = decode(inst, z["pop"], z["mask"], *penalty_weights(inst))
        over = sum(r.duration > inst.TL + 1e-9 for r in sol.routes)
        st["runs"] += 1
        st["feasible"] += sol.feasible
        st["runs_over"] += over > 0
        st["routes_over"] += over
        st["routes"] += len(sol.routes)
    return st


def read_polish(name: str, algo: str) -> list[dict]:
    """Per-seed rows of results/polish_<name>_<algo>.csv (polish_experiment.py)."""
    path = RESULTS / f"polish_{name}_{algo}.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k in ("instance", "algorithm", "currency", "skipped_day_sizes"):
                continue
            if v in ("True", "False"):
                r[k] = v == "True"
            else:
                r[k] = float(v)
    return rows


def dp_section(data: dict) -> list[str]:
    """Exact per-day DP routing (src/dp_router.py): complexity, polish gains,
    and whether polish narrows the SA-GA gap."""
    from dp_router import MAX_DP_POINTS
    md = ["## Dynamic-programming route polish", "",
          "`src/dp_router.py` is an opt-in layer (`runner.py --polish`, "
          "`exporter.py --polish`, `src/polish_experiment.py`). For a solution "
          "found by SA or GA it keeps the schedule, bins and waste fixed and "
          "replaces each day's greedy route split with the exact optimum: "
          "Held–Karp gives the best closed tour for every subset of the day's "
          "points, and a partition DP picks the cheapest set of at most nV "
          "routes within Q and TL. A day is replaced only if that is strictly "
          "shorter, so bin cost is unchanged and routing cost cannot rise. SA "
          "and GA themselves are unchanged, and the Appendix A validation never "
          "touches this code. A worked example by hand is in "
          "`docs/dp_dry_run.md`.", ""]

    # --- complexity ---------------------------------------------------------
    timing = {}
    tpath = RESULTS / "dp_timing.csv"
    if tpath.exists():
        with tpath.open() as fh:
            timing = {int(r["n_points"]): float(r["median_ms"])
                      for r in csv.DictReader(fh)}
    K = 3
    md += ["### Complexity against day size", "",
           f"For n points in a day and a fleet of K = {K}: Held–Karp does "
           "n²·2ⁿ steps and keeps an n·2ⁿ table; the partition DP looks at "
           "every (set, subset containing its first point) pair, (3ⁿ − 1)/2 of "
           "them, each for K route counts. Measured times are "
           "`dp_day_routes` on the Chennai matrix with 1 m³ per point "
           "(`polish_experiment.py --benchmark`, median of 5).", "",
           "| Points in the day | Held–Karp steps n²·2ⁿ | Partition steps K·(3ⁿ−1)/2 "
           "| DP tables (MB) | Measured time |",
           "|---|---|---|---|---|"]
    for n in (4, 6, 8, 10, 12, 13, 14, 15, 16, 18, 21):
        hk = n * n * 2 ** n
        part = K * (3 ** n - 1) // 2
        # dp (float64) + parent (int8) of n*2^n, f and choice of (K+1)*2^n
        mb = (n * 2 ** n * 9 + (K + 1) * 2 ** n * 16) / 1e6
        if n in timing:
            t = (f"{timing[n]:,.3f} ms" if timing[n] < 1000
                 else f"{timing[n] / 1000:,.2f} s")
        else:
            est = timing.get(16, 0) * 3 ** (n - 16) / 1000
            t = f"≈ {est:,.0f} s (extrapolated ×3 per point)" if timing else "—"
        mark = " ← MAX_DP_POINTS" if n == MAX_DP_POINTS else ""
        md.append(f"| {n}{mark} | {hk:,} | {part:,} | {mb:,.1f} | {t} |")
    md += ["", f"**Why the threshold is {MAX_DP_POINTS}:** each extra point "
           "multiplies the work by about 3. At 14 points a day takes under "
           "0.1 s, so polishing a whole week stays interactive; at 16 it is "
           "already about half a second per day.", "",
           "**Why not DP for the whole problem at 21 points:** routing alone "
           "for one 21-point day would need about 2·10¹⁰ partition steps "
           "(minutes per day) and over 0.5 GB of tables. That is too slow to "
           "sit inside SA or GA, which evaluate about 500,000 candidate "
           "schedules per run. Choosing the schedule as well is far beyond "
           "it: each of 21 points can be collected on any subset of 6 working "
           "days, 2¹²⁶ ≈ 10³⁸ schedules. Each schedule changes every day's "
           "loads through the waste accumulation (Eq. 2l) and every point's "
           "bin, so the problem does not split into independent days. DP is "
           "therefore used where it is exact and cheap: the routing of one "
           "day, once the metaheuristic has fixed the schedule.", ""]

    # --- polish gains -------------------------------------------------------
    names = [n for n in INSTANCES if read_polish(n, "sa")]
    if not names:
        return md
    md += ["### Polishing the saved best solutions", "",
           "The best solution of each of the 30 SA and 30 GA seeds, as saved by "
           "the runs above, polished without re-running SA or GA "
           "(`python src/polish_experiment.py`; per-seed results in "
           "`results/polish_<instance>_<algo>.csv`).", "",
           "| Instance | Algo | Mean routing gain | Best overall before → after "
           "| Seeds changed | Days improved | Mean DP time / day "
           f"| Days skipped (> {MAX_DP_POINTS} points) |",
           "|---|---|---|---|---|---|---|---|"]
    for n in names:
        m = lambda x, n=n: money(n, x)  # noqa: E731
        for a in ALGOS:
            rows = read_polish(n, a)
            if not rows:
                continue
            gain = np.mean([r["routing_improvement"] for r in rows])
            gpct = np.mean([r["routing_improvement_pct"] for r in rows])
            before = min(r["overall_cost"] for r in rows)
            after = min(r["polished_overall_cost"] for r in rows)
            changed = sum(r["days_improved"] > 0 for r in rows)
            days = sum(r["days"] for r in rows)
            days_imp = sum(r["days_improved"] for r in rows)
            skipped = sum(r["days_skipped"] for r in rows)
            ms = np.mean([r["dp_ms_per_day"] for r in rows])
            md.append(f"| {n} | {LABEL[a]} | {m(gain)} ({gpct:.2f}%) | "
                      f"{m(before)} → {m(after)} | {changed} / {len(rows)} | "
                      f"{int(days_imp)} / {int(days)} | {ms:.2f} ms | "
                      f"{int(skipped)} |")
    md.append("")
    if "i.12.1" in names:
        sa_best = min(r["polished_overall_cost"] for r in read_polish("i.12.1", "sa"))
        # the paper's own Appendix A solution, polished
        import validate_appendix as va
        from dp_router import polish_with_report
        app = inst_of("i.12.1")
        app_sol = decode(app, va.POP, va.MASK, *penalty_weights(app))
        _, app_rep = polish_with_report(app, app_sol)
        app_changed = sum(d.status == "improved" for d in app_rep.days)
        md += [f"On i.12.1 the best SA solution polishes to {sa_best:.2f} US$: "
               f"{pct(sa_best, PAPER_MILP)} relative to the MILP optimum of "
               f"{PAPER_MILP} (Table 10), and {pct(sa_best, PAPER_SA_MIN)} "
               f"relative to the paper's SA minimum of {PAPER_SA_MIN}. "
               + ("Polishing the paper's own Appendix A solution changes none "
                  f"of its {len(app_rep.days)} days: its published routes are "
                  "already optimal for its schedule." if app_changed == 0 else
                  f"Polishing the paper's own Appendix A solution improves "
                  f"{app_changed} of its {len(app_rep.days)} days."), ""]

    # --- SA+DP vs GA+DP -----------------------------------------------------
    md += ["### Does DP polish narrow the SA–GA gap?", "",
           "Mean overall cost of the 30 seeds, before and after polish, and a "
           "two-sided Mann-Whitney U test of SA+DP against GA+DP:", "",
           "| Instance | SA → SA+DP | GA → GA+DP | GA − SA gap before → after "
           "| Gap closed | U, p (SA+DP vs GA+DP) |",
           "|---|---|---|---|---|---|"]
    closed_share = []
    for n in names:
        m = lambda x, n=n: money(n, x)  # noqa: E731
        sa, ga = read_polish(n, "sa"), read_polish(n, "ga")
        if not sa or not ga:
            continue
        sa0 = np.mean([r["overall_cost"] for r in sa])
        sa1 = np.mean([r["polished_overall_cost"] for r in sa])
        ga0 = np.mean([r["overall_cost"] for r in ga])
        ga1 = np.mean([r["polished_overall_cost"] for r in ga])
        gap0, gap1 = ga0 - sa0, ga1 - sa1
        share = (gap0 - gap1) / gap0 if gap0 else 0.0
        closed_share.append(share)
        u, p = mannwhitneyu([r["polished_overall_cost"] for r in sa],
                            [r["polished_overall_cost"] for r in ga],
                            alternative="two-sided")
        md.append(f"| {n} | {m(sa0)} → {m(sa1)} | {m(ga0)} → {m(ga1)} | "
                  f"{m(gap0)} ({gap0 / sa0:.1%}) → {m(gap1)} ({gap1 / sa1:.1%}) | "
                  f"{share:.0%} | U = {u:.0f}, p = {p:.3g} |")
    bin_gap = []
    for n in names:
        sa, ga = read_polish(n, "sa"), read_polish(n, "ga")
        bin_gap.append((n, np.mean([r["bin_cost"] for r in sa]),
                        np.mean([r["bin_cost"] for r in ga])))
    md += ["", f"DP polish closes {min(closed_share):.0%}–{max(closed_share):.0%} "
           "of the gap. GA gains far more from polish than SA (6–11% of "
           "routing cost against under 0.3%). A likely reading: SA's "
           "exchange moves keep refining the visiting orders of a schedule, "
           "while GA's crossover recombines orders and leaves poor ones; that "
           "part of GA's deficit is route construction, which DP removes. "
           "The rest is the schedule itself (which days each point is "
           "collected, hence the bins and loads), which DP holds fixed. "
           "Mean bin cost, SA vs GA: "
           + "; ".join(f"{n} {money(n, b_sa)} vs {money(n, b_ga)}"
                       for n, b_sa, b_ga in bin_gap)
           + ". SA+DP still beats GA+DP on every instance, so the ranking "
           "stands, but the margin by which GA loses is roughly halved.", ""]
    return md


def pct(ours: float, ref: float) -> str:
    return f"{(ours - ref) / ref * 100:+.2f}%"


# ---------------------------------------------------------------------------
def convergence_plot(name: str) -> Path | None:
    """Median best fitness vs evaluations over all seeds, IQR shaded."""
    curves = {}
    for algo in ALGOS:
        runs = []
        for f in sorted((RESULTS / "solutions").glob(f"{name}_{algo}_seed*.npz")):
            with np.load(f) as z:
                if "history" in z:
                    runs.append((z["history_evals"], z["history"]))
        if runs:
            curves[algo] = runs
    if not curves:
        return None

    x_max = max(int(e[-1]) for runs in curves.values() for e, _ in runs)
    grid = np.linspace(0, x_max, 600)

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    y_lo, y_hi, final_q3, ends = np.inf, 0.0, 0.0, []
    for algo, runs in curves.items():
        # best-so-far is a step function: value of the last entry <= x
        mat = np.full((len(runs), grid.size), np.nan)
        for k, (ev, h) in enumerate(runs):
            idx = np.searchsorted(ev, grid, side="right") - 1
            ok = idx >= 0
            mat[k, ok] = h[idx[ok]]
            mat[k, grid > ev[-1]] = h[-1]
        with warnings.catch_warnings():      # GA has no value before its
            warnings.simplefilter("ignore")  # first generation is scored
            med = np.nanmedian(mat, axis=0)
            q1, q3 = np.nanpercentile(mat, [25, 75], axis=0)
        ax.fill_between(grid, q1, q3, color=COLOR[algo], alpha=0.15, lw=0)
        ax.plot(grid, med, color=COLOR[algo], lw=2,
                label=f"{LABEL[algo]} (median of {len(runs)} runs, IQR shaded)")
        end = (inst_of(name).money(med[-1], 0)
               if inst_of(name).currency == "INR" else f"{med[-1]:.1f}")
        ends.append((med[-1], f"{LABEL[algo]} {end}"))
        final_q3 = max(final_q3, q3[-1])
        y_lo = min(y_lo, np.nanmin(q1))
        # frame the part of the curve after the first 5% of the budget, where
        # penalised infeasible starts no longer dominate the scale
        y_hi = max(y_hi, np.nanmax(q3[grid >= 0.05 * x_max]))

    # penalised infeasible solutions can persist well past 5% of the budget
    # (e.g. with a tight shift limit); never let them stretch the axis beyond
    # 1.3x the final values
    y_hi = min(y_hi, 1.3 * final_q3)
    ax.set_ylim(y_lo - 0.02 * (y_hi - y_lo), y_hi)

    # end labels; push them apart when the curves finish close together
    ends.sort()
    close = len(ends) == 2 and ends[1][0] - ends[0][0] < 0.06 * (y_hi - y_lo)
    for k, (y, text) in enumerate(ends):
        dy = (-6 if k == 0 else 6) if close else 0
        ax.annotate(text, (grid[-1], y), xytext=(6, dy),
                    textcoords="offset points", va="center", color=INK,
                    fontsize=9)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Fitness evaluations", color=MUTED)
    inst = inst_of(name)
    ax.set_ylabel(f"Best fitness so far ({inst.currency_symbol}/week)", color=MUTED)
    if inst.currency == "INR":
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: "₹" + indian_grouping(v, 0)))
    ax.set_title(f"Convergence on {name}: SA vs GA at equal evaluations",
                 color=INK, loc="left", fontsize=11)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"convergence_{name}.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    data = {(n, a): read_rows(n, a) for n in INSTANCES for a in ALGOS}
    md = ["# Experiment summary", "",
          "Each run is one seed; SA and GA are matched on fitness-function "
          "evaluations (see `results/calibration_*.txt`). All costs are per "
          "week, in each instance's own currency:", ""]
    for n in INSTANCES:
        inst = inst_of(n)
        if inst.currency == "USD":
            md.append(f"- **{n}**: US dollars (US$), the paper's own cost "
                      "parameters.")
        else:
            md.append(f"- **{n}**: Indian rupees (₹). These figures are an FX "
                      "conversion of the paper's US$ cost parameters at "
                      f"{inst.fx_rate} ₹/US$, **not** independently sourced "
                      "Indian rates.")
    md.append("")

    # --- per instance / algorithm -----------------------------------------
    md += ["## Results by instance and algorithm", "",
           "| Instance | Algo | n | Min | Median | Mean | 95% CI | Mean evals "
           "| Mean runtime (s) | All feasible |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    stats = {}
    for (n, a), rows in data.items():
        if not rows:
            md.append(f"| {n} | {LABEL[a]} | 0 | — | — | — | — | — | — | — |")
            continue
        s = summarize(rows)
        stats[(n, a)] = s
        ev = np.mean([r["evaluations"] for r in rows])
        m = lambda x: money(n, x)  # noqa: E731
        md.append(f"| {n} | {LABEL[a]} | {s['n']} | {m(s['min'])} | "
                  f"{m(s['median'])} | {m(s['mean'])} | "
                  f"[{m(s['ci_low'])}, {m(s['ci_high'])}] | {ev:,.0f} | "
                  f"{s['mean_runtime_s']:.1f} | {'yes' if s['all_feasible'] else 'NO'} |")
    md.append("")

    # evaluation matching check
    md += ["Evaluation matching (GA vs mean SA evaluations):", ""]
    for n in INSTANCES:
        sa, ga = data[(n, "sa")], data[(n, "ga")]
        if sa and ga:
            e_sa = np.mean([r["evaluations"] for r in sa])
            e_ga = np.mean([r["evaluations"] for r in ga])
            lo = min(r["evaluations"] for r in sa)
            hi = max(r["evaluations"] for r in sa)
            md.append(f"- {n}: SA {e_sa:,.0f} (range {lo:,}–{hi:,}), "
                      f"GA {e_ga:,.0f} → {pct(e_ga, e_sa)}")
    md.append("")

    # --- paper comparison --------------------------------------------------
    md += ["## i.12.1 against the paper", "",
           "Paper: Table 9 (SA, min 188.0 / mean 193.7) and Table 10 "
           "(MILP optimum, overall 178.42). Both sides in US$.", "",
           "| Metric | Ours | Paper | Difference |", "|---|---|---|---|"]
    for a in ALGOS:
        s = stats.get(("i.12.1", a))
        if not s:
            continue
        md.append(f"| {LABEL[a]} min | {s['min']:.2f} | SA min {PAPER_SA_MIN:.1f} "
                  f"| {pct(s['min'], PAPER_SA_MIN)} |")
        md.append(f"| {LABEL[a]} mean | {s['mean']:.2f} | SA mean {PAPER_SA_MEAN:.1f} "
                  f"| {pct(s['mean'], PAPER_SA_MEAN)} |")
        md.append(f"| {LABEL[a]} min | {s['min']:.2f} | MILP {PAPER_MILP:.2f} "
                  f"| {pct(s['min'], PAPER_MILP)} |")
        md.append(f"| {LABEL[a]} mean | {s['mean']:.2f} | MILP {PAPER_MILP:.2f} "
                  f"| {pct(s['mean'], PAPER_MILP)} |")
    md.append("")

    # --- Mann-Whitney U ------------------------------------------------------
    md += ["## SA vs GA: Mann-Whitney U test (overall cost, two-sided)", "",
           "| Instance | n (SA, GA) | U | p-value | Median SA | Median GA | "
           "Verdict (α = 0.05) |", "|---|---|---|---|---|---|---|"]
    for n in INSTANCES:
        sa = [r["overall_cost"] for r in data[(n, "sa")]]
        ga = [r["overall_cost"] for r in data[(n, "ga")]]
        if len(sa) < 2 or len(ga) < 2:
            continue
        u, p = mannwhitneyu(sa, ga, alternative="two-sided")
        better = "SA" if np.median(sa) < np.median(ga) else "GA"
        verdict = (f"significant — {better} lower" if p < 0.05
                   else "no significant difference")
        md.append(f"| {n} | {len(sa)}, {len(ga)} | {u:.1f} | {p:.3g} | "
                  f"{money(n, np.median(sa))} | {money(n, np.median(ga))} | "
                  f"{verdict} |")
    md.append("")

    # --- Chennai cost split ------------------------------------------------
    for n in [k for k in INSTANCES if k.startswith("chennai")]:
        m = lambda x, n=n: money(n, x)  # noqa: E731
        md += [f"## {n}: cost breakdown ({inst_of(n).currency})", "",
               "| Algo | | Bin cost | Routing cost | Overall |",
               "|---|---|---|---|---|"]
        for a in ALGOS:
            rows = data[(n, a)]
            if not rows:
                continue
            best = min(rows, key=lambda r: r["overall_cost"])
            for label, f in (("mean", np.mean), ("median", np.median)):
                md.append(f"| {LABEL[a]} | {label} | "
                          f"{m(f([r['bin_cost'] for r in rows]))} | "
                          f"{m(f([r['routing_cost'] for r in rows]))} | "
                          f"{m(f([r['overall_cost'] for r in rows]))} |")
            md.append(f"| {LABEL[a]} | best (seed {best['seed']}) | "
                      f"{m(best['bin_cost'])} | {m(best['routing_cost'])} | "
                      f"{m(best['overall_cost'])} |")
        md.append("")

    # --- free-flow vs peak ------------------------------------------------
    scen = []
    for label, n in SCENARIOS:
        if data.get((n, "sa")):
            b = best_solution(n)
            if b is not None:
                scen.append((label, n, b[0], b[1], inst_of(n)))
    if len(scen) >= 2:
        mon = scen[0][4].money

        def longest(sol):
            return max(r.duration for r in sol.routes)

        def over(sol, tl):
            return sum(r.duration > tl + 1e-9 for r in sol.routes)

        def row(label, values, fmt, change=True):
            cells = []
            for k, v in enumerate(values):
                c = fmt(v)
                if change and k > 0 and values[0]:
                    c += f" ({pct(v, values[0])})"
                cells.append(c)
            md.append(f"| {label} | " + " | ".join(cells) + " |")

        sols = [x[3] for x in scen]
        insts = [x[4] for x in scen]
        names = [x[1] for x in scen]
        stats = [run_stats(n) for n in names]
        md += ["## Chennai: free-flow vs peak-hour traffic", "",
               "The peak scenarios multiply every OSRM travel time by "
               f"{json_traffic('chennai_guindy_peak')}. All use the same points, "
               "depot, bins and rupee cost parameters. Under Eq. (10) the shift "
               "limit TL is derived from the travel-time matrix, so in the plain "
               f"peak scenario it grows with traffic ({insts[0].TL:.0f} → "
               f"{insts[1].TL:.0f} min). The third scenario holds TL at the "
               "free-flow 96 min (`load_instance(..., TL=96)`, "
               "`runner.py --tl 96`), so congestion has to be absorbed within "
               "the original shift.", "",
               "Best solution per scenario ("
               + ", ".join(f"`{x[2]}`" for x in scen)
               + "), the 30-run SA mean, and feasibility over all 60 runs "
               "(30 SA + 30 GA). Changes in brackets are relative to free-flow.",
               "",
               "| | " + " | ".join(x[0] for x in scen) + " |",
               "|---" * (len(scen) + 1) + "|"]
        row("Shift limit TL (min)", [i.TL for i in insts], lambda x: f"{x:.0f}")
        row("Overall cost (best)", [x.overall_cost for x in sols], mon)
        row("Bin cost (best)", [x.bin_cost for x in sols], mon)
        row("Routing cost (best)", [x.routing_cost for x in sols], mon)
        row("Overall cost (SA mean of 30)",
            [float(np.mean([r["overall_cost"] for r in data[(n, "sa")]]))
             for n in names], mon)
        row("Routes per week (best)", [len(x.routes) for x in sols],
            lambda x: f"{x}")
        row("Total route time (min, best)",
            [sum(r.duration for r in x.routes) for x in sols],
            lambda x: f"{x:.1f}")
        row("Longest route (min, best)", [longest(x) for x in sols],
            lambda x: f"{x:.1f}")
        row("Longest route / TL (best)",
            [longest(x) / i.TL for x, i in zip(sols, insts)],
            lambda x: f"{x:.0%}", change=False)
        row("Routes over TL (best)",
            [over(x, i.TL) for x, i in zip(sols, insts)],
            lambda x: f"{x}", change=False)
        row("Penalty (best)", [x.penalty for x in sols], mon, change=False)
        row("Feasible (best)", [x.feasible for x in sols],
            lambda x: "yes" if x else "NO", change=False)
        row("Feasible runs (SA + GA)",
            [f"{st['feasible']} / {st['runs']}" for st in stats],
            lambda x: x, change=False)
        row("Runs with a route over TL",
            [f"{st['runs_over']} / {st['runs']}" for st in stats],
            lambda x: x, change=False)
        row("Routes over TL, all runs",
            [f"{st['routes_over']} of {st['routes']}" for st in stats],
            lambda x: x, change=False)
        md.append("")

        for k in range(1, len(scen)):
            same = bool((sols[0].bins == sols[k].bins).all())
            diff = int((sols[0].bins != sols[k].bins).sum())
            md.append(f"- {scen[k][0]}: bin combinations "
                      + ("identical to free-flow." if same else
                         f"differ from free-flow at {diff} of "
                         f"{insts[0].n_points} points."))
        if len(scen) == 3:
            peak, pinned, ip = sols[1], sols[2], insts[2]
            gamma = penalty_weights(ip)[1]
            md.append(
                f"- The shift-length penalty (Eq. 7) is γ × minutes over the "
                f"limit, here {ip.money(gamma, 0)} per minute (γ = 1000 US$, "
                f"converted), about {gamma / ip.ccv:,.0f}× the cost of a minute "
                "of route time. Any route over TL therefore dominates the "
                "fitness, and the optimisers treat TL as a hard constraint.")
            pinned_bad = [r for a in ALGOS for r in data[(names[2], a)]
                          if not r["feasible"]]
            for r in pinned_bad:
                md.append(
                    f"- Infeasible run: {r['algorithm']} seed {r['seed']} ended "
                    f"with {r.get('routes_over_tl', '?')} route(s) over "
                    f"{ip.TL:.0f} min (longest {r.get('longest_route_min', '?')} "
                    f"min). Its penalty is {ip.money(r.get('penalty', 0))}, on "
                    f"top of a {ip.money(r['overall_cost'])} plan, a fitness of "
                    f"{ip.money(r['overall_cost'] + r.get('penalty', 0))}: that "
                    "run never found a feasible plan cheaper than this "
                    "penalised one.")
            md += ["", f"**Does holding the shift at {ip.TL:.0f} min bind?** The "
                   "plain-peak runs, re-checked against the pinned limit, and "
                   "plain vs pinned overall cost (two-sided Mann-Whitney U):", "",
                   f"| Algo | Plain-peak runs that would break {ip.TL:.0f} min "
                   "| Longest plain-peak route (min) | Median cost plain → pinned "
                   "| p-value |", "|---|---|---|---|---|"]
            for a in ALGOS:
                over_runs, worst = 0, 0.0
                for f in solution_files(names[1]):
                    if f"_{a}_seed" not in f.name:
                        continue
                    with np.load(f) as z:
                        sol = decode(ip, z["pop"], z["mask"], *penalty_weights(ip))
                    over_runs += over(sol, ip.TL) > 0
                    worst = max(worst, longest(sol))
                plain = [r["overall_cost"] for r in data[(names[1], a)]]
                pin = [r["overall_cost"] for r in data[(names[2], a)]]
                _, p = mannwhitneyu(plain, pin, alternative="two-sided")
                md.append(f"| {LABEL[a]} | {over_runs} / {len(plain)} | "
                          f"{worst:.1f} | {mon(np.median(plain))} → "
                          f"{mon(np.median(pin))} ({pct(np.median(pin), np.median(plain))}) "
                          f"| {p:.3g} |")
            md.append("")
        md.append("")

    md += dp_section(data)

    # --- figures -----------------------------------------------------------
    md += ["## Convergence", "",
           "Median best fitness over all seeds against evaluations spent; the "
           "shaded band is the interquartile range. A run that stopped early "
           "holds its final value. The y-axis leaves out penalised infeasible "
           "solutions (it starts after the first 5% of the budget and is capped "
           "at 1.3× the final values); where a curve enters from above the "
           "frame, the median run was still infeasible.",
           ""]
    for n in INSTANCES:
        out = convergence_plot(n)
        if out:
            md.append(f"![Convergence on {n}](figures/{out.name})")
            md.append("")

    # --- notes -------------------------------------------------------------
    md += ["## Notes for reading these results", "",
           "- The MILP value (Table 10) is the proven optimum for i.12.1, so the "
           "\"vs MILP\" rows are optimality gaps. Only SA and MILP figures from "
           "the paper are compared here.",
           "- A Mann-Whitney U of 0 means complete separation: every SA run "
           "beat every GA run. With 30 runs each, every instance with complete "
           "separation gets the same p-value (normal approximation).",
           "- Evaluations are matched, wall time is not. GA's crossover, "
           "mutation and repair are pure Python, while the decoder is compiled, "
           "so GA takes ~5× longer for the same number of evaluations. "
           "Runtimes were measured with 11 runs in parallel, which is about "
           "4× slower per run than running alone.",
           "- SA's evaluation count varies by seed because it stops after 100 "
           "non-improving temperature steps. GA's generations were set from "
           "the 30-run SA mean (see `overnight_log.md`).",
           "- The Chennai runs (2026-10-02) ran on battery power, so the CPU "
           "was throttled by varying amounts. Their runtimes are not "
           "comparable with i.12.1's or with each other; evaluation counts "
           "are the fair measure of effort."]

    md += ["- Rupee figures are the paper's US$ cost parameters converted at "
           f"{inst_of('chennai_guindy').fx_rate} ₹/US$ (mid-market, "
           f"{meta_of('chennai_guindy').get('fx_date')}, "
           f"{meta_of('chennai_guindy').get('fx_source')}). The penalty weights "
           "and SA's final temperature are converted too, so the optimisation "
           "problem is the same as in US$. Locally sourced Greater Chennai "
           "Corporation rates would change the bin-vs-routing tradeoff, not just "
           "the totals.", ""]

    (RESULTS / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS / 'summary.md'}")


if __name__ == "__main__":
    main()
