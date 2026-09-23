# Integrated Periodic VRP + Capacitated Facility Location — Chennai Application

Implementation of the model in Gonzalez, Rossit, Frutos & Mendez (2025),
*Annals of Operations Research* 350:979–1015, applied to a collection-point
network around CEG Guindy, Anna University, Chennai.

## Status

| Module | File | Status |
|---|---|---|
| M3 Loader | `src/loader.py` | done |
| M4 Decoder / evaluator | `src/decoder.py` | done |
| M8 Validation harness | `src/validate_appendix.py` | **done — 7/7 pass** |
| M5 SA + GA | `src/optimizers.py` | done |
| M1 Chennai instance | `src/instance_builder.py` | done (21 points) |
| M2 OSRM travel matrix | `src/instance_builder.py` | done (OSRM + fallback) |
| M6 Experiment runner | `src/runner.py` | done |
| M7 Exporter | `src/exporter.py` | pending |
| M9 Leaflet UI | `ui/index.html` | pending |

## Validation

`python src/validate_appendix.py` feeds the chromosome published in the
paper's Table 16 into our decoder and checks the result against the paper's
Tables 16 and 17:

- waste accumulation `w[i,t]` — max deviation 0.0000 m³
- bin combination per point — exact match on all 12 points
- route stop sequences — exact match on all 10 routes
- route durations — max deviation 0.0000 min
- capacity and feasibility — all satisfied

This is the correctness anchor for the whole project: the same engine is then
pointed at the Chennai instance without modification.

## Note on the published bin-cost table

The paper's Table 2 lists a weekly cost of 1.56 US$ for bin combinations 1–7,
which appears to be a typesetting error (the values are identical down the
column). The authors' own dataset (`containers.txt`) gives the real figures:

| Bin | Capacity (m³) | Service (min) | Weekly cost (US$) |
|---|---|---|---|
| 0 | 1.1 | 0.70 | 0.78 |
| 1 | 2.2 | 1.40 | 1.56 |
| 2 | 2.4 | 0.66 | 2.23 |
| 3 | 3.3 | 2.10 | 2.34 |
| 4 | 3.5 | 1.36 | 3.00 |
| 5 | 4.3 | 1.37 | 3.37 |
| 6 | 4.8 | 1.32 | 4.45 |
| 7 | 5.6 | 1.33 | 4.82 |

We use the dataset values. This is worth a sentence in the report.

## A modelling detail worth defending in the viva

Bin selection is **not** "cheapest bin that fits". A cheaper bin can have a
longer service time, and service time is paid on every visit at 0.5764 US$/min.
The correct per-point objective is

    CIN_b + CCV · S_b · (visits that week)

Point 9 in the Appendix A example proves it: bin 3 (3.3 m³, 2.34 US$) fits its
3.16 m³ peak and is cheaper than bin 4 (3.5 m³, 3.00 US$), but bin 3 is
0.74 min slower per visit — over 4 visits that costs 1.71 US$, outweighing the
0.66 US$ saving. The paper picks bin 4, and so do we.

## Run

```bash
pip install numpy
python src/validate_appendix.py
```

## Data

Instances from https://github.com/diegorossit/ANOR-S-24-01950


## Chennai instance

`python src/instance_builder.py` builds `data/chennai_guindy/` — 21 real
collection points along the Sardar Patel Road / Gandhi Mandapam Road / Adyar /
Saidapet / Velachery corridor, in the paper's own file format.

- **Travel times**: OSRM road routing, with a haversine + detour-factor
  fallback that is recorded in `meta.json` so it never silently ends up in
  your results. Run without `--offline` on your laptop to get real road times.
- **Waste generation**: estimated from GCC figures (0.71 kg/person/day,
  0.40 t/m³ loose density) times each point's catchment population — not
  copied from the paper.
- **Depot**: currently a PLACEHOLDER. Replace with the real GCC transfer
  station for this zone before the results go in the report.

### A finding worth a paragraph in the report

The first build was infeasible, and the reason is a genuine result rather
than a bug. Because Sunday is a drivers' rest day, every collection point
holds **two** days of waste on Monday morning. So a single bin combination can
only serve a point generating up to

    max bin capacity / (1 + rest days) = 5.6 / 2 = 2.80 m³/day

The paper's bin catalogue is sized for Bahía Blanca, where points generate
1.0–1.6 m³/day. Chennai points with large catchments break it. There are three
legitimate responses — split the catchment, extend `containers.txt` with the
larger skip bins used in Indian cities, or collect on the rest day — and
`check_bin_feasibility()` now flags the problem at build time. We took the
first option, using realistic per-bin-point catchments of 240–700 residents.

## Running experiments

```bash
python src/runner.py --instance data/chennai_guindy --algo both --runs 10
python src/runner.py --instance data/12_1 --name i.12.1 --algo both --runs 30
```

Runs are parallel across cores and checkpointed to `results/*.csv` after every
seed. **Increase `--iters-per-temp`**: the paper used ~1.7 million fitness
evaluations per SA run, and the smoke tests here used ~50,000, which is why
seed-to-seed spread is still wide. On a multi-core laptop, use
`--iters-per-temp 5000` or higher.
