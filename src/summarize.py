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
             "chennai_guindy_peak": ROOT / "data" / "chennai_guindy_peak"}
SCENARIOS = [("Free-flow", "chennai_guindy"), ("Peak (×1.5)", "chennai_guindy_peak")]
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
    return sorted(rows, key=lambda r: r["seed"])


_INST: dict[str, Instance] = {}


def inst_of(name: str) -> Instance:
    """The instance behind a runner.py name -- for its currency and TL."""
    if name not in _INST:
        _INST[name] = load_instance(INSTANCES[name], name)
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
    for f in sorted((RESULTS / "solutions").glob(f"{name}_*_seed*.npz")):
        with np.load(f) as z:
            sol = decode(inst, z["pop"], z["mask"], *penalty_weights(inst))
        key = (not sol.feasible, sol.fitness)
        if best is None or key < best[0]:
            best = (key, f.stem, sol)
    return None if best is None else (best[1], best[2])


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
    y_lo, y_hi = np.inf, 0.0
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
        ax.annotate(f"{LABEL[algo]} {end}", (grid[-1], med[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=9)
        y_lo = min(y_lo, np.nanmin(q1))
        # frame the part of the curve after the first 5% of the budget, where
        # penalised infeasible starts no longer dominate the scale
        y_hi = max(y_hi, np.nanmax(q3[grid >= 0.05 * x_max]))

    ax.set_ylim(y_lo - 0.02 * (y_hi - y_lo), y_hi)
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

    # --- free-flow vs peak --------------------------------------------------
    scen = [(label, n, best_solution(n)) for label, n in SCENARIOS
            if (n, "sa") in data and data[(n, "sa")]]
    scen = [(label, n, b) for label, n, b in scen if b is not None]
    if len(scen) == 2:
        (la, na, (fa, sa_)), (lb, nb, (fb, sb_)) = scen
        ia, ib = inst_of(na), inst_of(nb)

        def longest(sol):
            r = max(sol.routes, key=lambda r: r.duration)
            return r.duration

        def row(label, a, b, fmt):
            diff = pct(b, a) if a else "—"
            md.append(f"| {label} | {fmt(a)} | {fmt(b)} | {diff} |")

        mon = lambda x: ia.money(x)  # noqa: E731
        md += ["## Chennai: free-flow vs peak-hour traffic", "",
               f"The peak scenario multiplies every OSRM travel time by "
               f"{json_traffic(nb)}. Both use the same points, depot, bins and "
               f"rupee cost parameters. Under Eq. (10) the shift limit TL is "
               f"derived from the travel-time matrix, so it grows with traffic too "
               f"({ia.TL:.0f} → {ib.TL:.0f} min).", "",
               f"Best solution per scenario (`{fa}` and `{fb}`), plus the 30-run "
               "SA mean:", "",
               f"| | {la} | {lb} | Change |", "|---|---|---|---|"]
        row("Overall cost (best)", sa_.overall_cost, sb_.overall_cost, mon)
        row("Bin cost (best)", sa_.bin_cost, sb_.bin_cost, mon)
        row("Routing cost (best)", sa_.routing_cost, sb_.routing_cost, mon)
        row("Overall cost (SA mean of 30)",
            float(np.mean([r["overall_cost"] for r in data[(na, "sa")]])),
            float(np.mean([r["overall_cost"] for r in data[(nb, "sa")]])), mon)
        row("Routes per week", len(sa_.routes), len(sb_.routes), lambda x: f"{x}")
        row("Total route time (min)", sum(r.duration for r in sa_.routes),
            sum(r.duration for r in sb_.routes), lambda x: f"{x:.1f}")
        row("Longest route (min)", longest(sa_), longest(sb_), lambda x: f"{x:.1f}")
        row("Shift limit TL (min)", ia.TL, ib.TL, lambda x: f"{x:.0f}")
        md.append(f"| Longest route / TL | {longest(sa_) / ia.TL:.0%} | "
                  f"{longest(sb_) / ib.TL:.0%} | |")
        md.append(f"| Feasible | {'yes' if sa_.feasible else 'NO'} | "
                  f"{'yes' if sb_.feasible else 'NO'} | |")
        same_bins = bool((sa_.bins == sb_.bins).all())
        md += ["", f"Bin combinations are {'identical' if same_bins else 'different'} "
               "in the two best solutions"
               + ("" if same_bins else
                  f" ({int((sa_.bins != sb_.bins).sum())} of {ia.n_points} points "
                  "differ)") + ".", ""]

    # --- figures -----------------------------------------------------------
    md += ["## Convergence", "",
           "Median best fitness over all seeds against evaluations spent; the "
           "shaded band is the interquartile range. A run that stopped early "
           "holds its final value. The y-axis starts after the first 5% of the "
           "budget, where penalised infeasible starts would flatten the scale.",
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
