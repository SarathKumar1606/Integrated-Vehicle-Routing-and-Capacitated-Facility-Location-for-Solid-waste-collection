# Dynamic-programming polish: a worked example by hand

This walks through `src/dp_router.py` on a toy instance small enough to do
with pen and paper. Every number below is reproduced by the real code in
`tests/test_dp_dry_run.py`. The DP does two things for one day's points:

1. **Held–Karp**: for every subset of the day's points, the shortest closed
   tour depot → subset → depot.
2. **Partition DP**: the cheapest way to split the day's points into at most
   `nV` routes, each within the vehicle capacity Q and the shift limit TL.

The metaheuristic (SA or GA) still decides which points are collected on which
day, and therefore the bins and the loads. DP only replaces the decoder's
greedy route split for that fixed assignment.

---

## 1. The toy instance

Four collection points A–D, a 3-day week, day 3 is the rest day.

| Point | Daily waste W (m³) |
|---|---|
| A | 1.0 |
| B | 1.5 |
| C | 0.8 |
| D | 1.2 |

Travel times in minutes (symmetric; 0 is the depot):

|   | 0 | A | B | C | D |
|---|---|---|---|---|---|
| **0** | – | 5 | 7 | 6 | 8 |
| **A** | 5 | – | 3 | 4 | 6 |
| **B** | 7 | 3 | – | 5 | 4 |
| **C** | 6 | 4 | 5 | – | 3 |
| **D** | 8 | 6 | 4 | 3 | – |

| Bin option | Capacity | Weekly cost | Service time per visit |
|---|---|---|---|
| small | 2.0 m³ | ₹100 | 1.0 min |
| large | 3.5 m³ | ₹160 | 1.5 min |

Vehicle: Q = 4 m³, CCV = ₹50 per minute, unloading TU = 5 min per route.

Two choices are not in the brief, and are ours for the example:

- **Fleet nV = 2.** Eq. (9) would give ⌈4/10⌉ = 1 truck, which cannot carry a
  day of this week's waste.
- **Shift limit TL = 30 min.** Tight enough that the shift limit, not only
  the capacity, rules some routes out.

**The collection schedule** (what SA/GA would have chosen):

| Point | Day 1 | Day 2 | Day 3 (rest) |
|---|---|---|---|
| A | – | ✓ | – |
| B | ✓ | ✓ | – |
| C | ✓ | ✓ | – |
| D | ✓ | ✓ | – |

For the greedy decoder the visiting order on both days is A, B, C, D.

---

## 2. Waste in each bin, and the bins

Waste is generated every day, including the rest day, and a visit empties the
bin at the end of the day. The week repeats, so we want the steady state.

- **A** (collected day 2 only): emptied at the end of day 2, so it holds
  1.0 at the end of day 3, 2.0 at the end of day 1 and 3.0 at the end of day 2,
  when it is collected. w = (2.0, **3.0**, 1.0).
- **B, C, D** (collected days 1 and 2): day 3 is not collected, so day 1
  collects two days' waste (day 3 + day 1) and day 2 collects one.

| Point | End of day 1 | End of day 2 | End of day 3 | Peak |
|---|---|---|---|---|
| A | 2.0 | **3.0** | 1.0 | 3.0 |
| B | **3.0** | **1.5** | 1.5 | 3.0 |
| C | **1.6** | **0.8** | 0.8 | 1.6 |
| D | **2.4** | **1.2** | 1.2 | 2.4 |

Bold = collected that day, which is the load the truck picks up.

**Bins.** A bin must hold the peak. The small bin (2.0 m³) is too small for
A, B and D, so they get the large one. C fits either; with 2 visits a week the
small bin costs 100 + 50 × 1.0 × 2 = ₹200 against 160 + 50 × 1.5 × 2 = ₹310
for the large, so C gets the small bin.

| Point | Bin | Service time | Visits |
|---|---|---|---|
| A | large | 1.5 min | 1 |
| B | large | 1.5 min | 2 |
| C | small | 1.0 min | 2 |
| D | large | 1.5 min | 2 |

Bin cost = 160 + 160 + 100 + 160 = **₹580 per week**. The DP never changes
this.

---

## 3. What the greedy decoder does on day 2

Day 2 collects A 3.0, B 1.5, C 0.8, D 1.2 m³ (6.5 m³ in total, so at least two
trucks). The decoder fills a truck in the visiting order A, B, C, D and cuts
when the next point would overflow it:

- A: 3.0. Adding B would make 4.5 > 4, so cut. **Route 1 = A.**
- B, C, D: 1.5 + 0.8 + 1.2 = 3.5 ≤ 4. **Route 2 = B → C → D.**

A **route's duration** (Eq. 1) is its travel time plus the service time at
each stop plus the unload time:

- Route 1: 0→A→0 = 5 + 5 = 10, + 1.5 + 5 = **16.5 min**
- Route 2: 0→B→C→D→0 = 7 + 5 + 3 + 8 = 23, + (1.5 + 1.0 + 1.5) + 5 = **32 min**

Route 2 is **2 min over the 30-minute limit**. The visiting order, not the
choice of points, is at fault: the same three points in another order fit,
as Held–Karp shows next.

---

## 4. Held–Karp on day 2

Write a subset as its letters, e.g. `BD` = {B, D}. In the code it is a bitmask
with A = 1, B = 2, C = 4, D = 8, so `BD` = 2 + 8 = 10.

**dp[S][j]** = the shortest travel time that leaves the depot, visits exactly
the points in S, and ends at j ∈ S (service and unloading are added later).

- **Start:** dp[{j}][j] = time from the depot to j.
- **Step:** dp[S][k] = min over j in S−{k} of ( dp[S−{k}][j] + time(j, k) ).

The code fills the table in increasing bitmask order. On a tie it keeps the
first candidate it found; "from" names the point visited just before.

### Single points

| S | end | dp | from |
|---|---|---|---|
| A | A | 5 | depot |
| B | B | 7 | depot |
| C | C | 6 | depot |
| D | D | 8 | depot |

### Pairs: dp[{X, Y}][Y] = dp[{X}][X] + time(X, Y)

| S | end A | end B | end C | end D |
|---|---|---|---|---|
| AB | 7 + 3 = **10** (B) | 5 + 3 = **8** (A) | | |
| AC | 6 + 4 = **10** (C) | | 5 + 4 = **9** (A) | |
| AD | 8 + 6 = **14** (D) | | | 5 + 6 = **11** (A) |
| BC | | 6 + 5 = **11** (C) | 7 + 5 = **12** (B) | |
| BD | | 8 + 4 = **12** (D) | | 7 + 4 = **11** (B) |
| CD | | | 8 + 3 = **11** (D) | 6 + 3 = **9** (C) |

### Triples: every candidate written out

To end at k, take the pair S−{k}, end it at j, and add the step j → k.

| S | end at | candidates | dp | from |
|---|---|---|---|---|
| ABC | A | via B: 11 + 3 = 14; via C: 12 + 4 = 16 | **14** | B |
| ABC | B | via A: 10 + 3 = 13; via C: 9 + 5 = 14 | **13** | A |
| ABC | C | via A: 10 + 4 = 14; via B: 8 + 5 = 13 | **13** | B |
| ABD | A | via B: 12 + 3 = 15; via D: 11 + 6 = 17 | **15** | B |
| ABD | B | via A: 14 + 3 = 17; via D: 11 + 4 = 15 | **15** | D |
| ABD | D | via A: 10 + 6 = 16; via B: 8 + 4 = 12 | **12** | B |
| ACD | A | via C: 11 + 4 = 15; via D: 9 + 6 = 15 | **15** | C (tie, first) |
| ACD | C | via A: 14 + 4 = 18; via D: 11 + 3 = 14 | **14** | D |
| ACD | D | via A: 10 + 6 = 16; via C: 9 + 3 = 12 | **12** | C |
| BCD | B | via C: 11 + 5 = 16; via D: 9 + 4 = 13 | **13** | D |
| BCD | C | via B: 12 + 5 = 17; via D: 11 + 3 = 14 | **14** | D |
| BCD | D | via B: 11 + 4 = 15; via C: 12 + 3 = 15 | **15** | B (tie, first) |

### All four

| end at | candidates (from the triple without k) | dp | from |
|---|---|---|---|
| A | via B: 13 + 3 = 16; via C: 14 + 4 = 18; via D: 15 + 6 = 21 | **16** | B |
| B | via A: 15 + 3 = 18; via C: 14 + 5 = 19; via D: 12 + 4 = 16 | **16** | D |
| C | via A: 15 + 4 = 19; via B: 15 + 5 = 20; via D: 12 + 3 = 15 | **15** | D |
| D | via A: 14 + 6 = 20; via B: 13 + 4 = 17; via C: 13 + 3 = 16 | **16** | C |

### Closing each subset into a route

closed(S) = min over j of ( dp[S][j] + time(j, depot) ) + service(S) + 5.

| S | best way home | travel | service | closed(S) | load (m³) |
|---|---|---|---|---|---|
| A | A: 5 + 5 | 10 | 1.5 | **16.5** | 3.0 |
| B | B: 7 + 7 | 14 | 1.5 | **20.5** | 1.5 |
| C | C: 6 + 6 | 12 | 1.0 | **18.0** | 0.8 |
| D | D: 8 + 8 | 16 | 1.5 | **22.5** | 1.2 |
| AB | A: 10 + 5 = 15; B: 8 + 7 = 15 | 15 | 3.0 | **23.0** | 4.5 |
| AC | A: 10 + 5 = 15; C: 9 + 6 = 15 | 15 | 2.5 | **22.5** | 3.8 |
| AD | A: 14 + 5 = 19; D: 11 + 8 = 19 | 19 | 3.0 | **27.0** | 4.2 |
| BC | B: 11 + 7 = 18; C: 12 + 6 = 18 | 18 | 2.5 | **25.5** | 2.3 |
| BD | B: 12 + 7 = 19; D: 11 + 8 = 19 | 19 | 3.0 | **27.0** | 2.7 |
| CD | C: 11 + 6 = 17; D: 9 + 8 = 17 | 17 | 2.5 | **24.5** | 2.0 |
| ABC | A: 14+5=19; B: 13+7=20; C: 13+6=19 | 19 | 4.0 | **28.0** | 5.3 |
| ABD | A: 15+5=20; B: 15+7=22; D: 12+8=20 | 20 | 4.5 | **29.5** | 5.7 |
| ACD | A: 15+5=20; C: 14+6=20; D: 12+8=20 | 20 | 4.0 | **29.0** | 5.0 |
| BCD | B: 13+7=20; C: 14+6=20; D: 15+8=23 | 20 | 4.0 | **29.0** | 3.5 |
| ABCD | A: 16+5=21; B: 16+7=23; C: 15+6=21; D: 16+8=24 | 21 | 5.5 | **31.5** | 6.5 |

`BCD` closes in **29 min** by ending at B. Reading the "from" column
backwards gives the tour: B ← D ← C, i.e. **depot → C → D → B → depot**
(6 + 3 + 4 + 7 = 20). The greedy order B → C → D took 23 min of travel.

### Which subsets can be a route?

A subset is **admissible** if its load ≤ Q = 4 **and** closed(S) ≤ TL = 30.

| S | load ≤ 4? | closed ≤ 30? | admissible |
|---|---|---|---|
| A, B, C, D | yes | yes | ✓ |
| AB | 4.5 ✗ | | ✗ |
| AC | 3.8 | 22.5 | ✓ |
| AD | 4.2 ✗ | | ✗ |
| BC, BD, CD | yes | yes | ✓ |
| ABC, ABD, ACD | over 4 ✗ | | ✗ |
| BCD | 3.5 | 29.0 | ✓ |
| ABCD | 6.5 ✗ | 31.5 ✗ | ✗ |

---

## 5. The partition DP on day 2

**f(S, k)** = the least total route time that serves exactly the points in S
with exactly k routes.

- f(∅, 0) = 0.
- f(S, k) = min over admissible T ⊆ S that **contain the first point of S** of
  ( closed(T) + f(S − T, k − 1) ).

Insisting that T contains S's first point (A before B before C before D)
makes each partition appear once: {A}+{BCD} and {BCD}+{A} are the same plan.

With **k = 1**, T must be S itself, so f(S, 1) = closed(S) when S is
admissible and ∞ otherwise.

With **k = 2**, T contains the first point of S and the rest goes on one more
route:

| S | candidates T (first point in bold) | f(S, 1) | f(S, 2) | best with ≤ 2 |
|---|---|---|---|---|
| A, B, C, D | only T = S, nothing left for route 2 | closed | ∞ | closed |
| AB | **A** + B: 16.5 + 20.5 = 37 | ∞ | **37** | 37 |
| AC | **A** + C: 16.5 + 18 = 34.5 | 22.5 | 34.5 | **22.5** |
| AD | **A** + D: 16.5 + 22.5 = 39 | ∞ | **39** | 39 |
| BC | **B** + C: 20.5 + 18 = 38.5 | 25.5 | 38.5 | **25.5** |
| BD | **B** + D: 20.5 + 22.5 = 43 | 27.0 | 43 | **27.0** |
| CD | **C** + D: 18 + 22.5 = 40.5 | 24.5 | 40.5 | **24.5** |
| ABC | **A** + BC: 16.5 + 25.5 = 42; **A**C + B: 22.5 + 20.5 = 43 | ∞ | **42** | 42 |
| ABD | **A** + BD: 16.5 + 27 = 43.5 | ∞ | **43.5** | 43.5 |
| ACD | **A** + CD: 16.5 + 24.5 = 41; **A**C + D: 22.5 + 22.5 = 45 | ∞ | **41** | 41 |
| BCD | **B** + CD: 45; **B**C + D: 48; **B**D + C: 45 | 29.0 | 45 | **29.0** |
| ABCD | see below | ∞ | **45.5** | 45.5 |

**The answer, f(ABCD, ·).** One route is impossible (6.5 m³). With two, T
must contain A and be admissible, so T is A or AC:

- T = **A**: closed(A) + f(BCD, 1) = 16.5 + 29.0 = **45.5**
- T = **AC**: closed(AC) + f(BD, 1) = 22.5 + 27.0 = 49.5

So day 2's optimum is **45.5 min: route A, and route C → D → B**. Every
other split is either over capacity (AB, AD, ABC, …) or over the shift
limit (ABCD).

---

## 6. Day 1, and the whole week

Day 1 collects B 3.0, C 1.6, D 2.4 (7.0 m³). The same two steps give
closed(B) = 20.5, closed(CD) = 24.5 (load exactly 4.0) and closed(BC) = 25.5,
closed(BD) = 27, closed(BCD) = 29. BC (4.6), BD (5.4) and BCD (7.0) are over
capacity, so the only two-route plan is **B + CD = 45.0 min**. The greedy
decoder already finds it (B, then C → D), so day 1 is left unchanged.

| | Greedy decoder | After DP polish |
|---|---|---|
| Day 1 routes | B (20.5) · C→D (24.5) | unchanged |
| Day 2 routes | A (16.5) · B→C→D (**32.0**, 2 min over TL) | A (16.5) · C→D→B (29.0) |
| Route minutes | 45.0 + 48.5 = 93.5 | 45.0 + 45.5 = 90.5 |
| Routing cost (₹50/min) | ₹4,675 | **₹4,525** |
| Bin cost | ₹580 | ₹580 (same) |
| Overall cost | ₹5,255 | **₹5,105** |
| Feasible | no: route over TL, penalty γ × 2 = 2,000 | **yes** |

The polish saves 3 minutes (₹150 a week), and it repairs the shift-limit
violation without touching the schedule or the bins.

---

## 7. The same numbers from the code

`tests/test_dp_dry_run.py` builds this instance as an `Instance` and runs
the real decoder and DP on it. It asserts every value in this document:
the waste table, the bins, the greedy routes and penalty, all 32 Held–Karp
entries, the 15 `closed` values, the partition table (best total with
≤ 1 and ≤ 2 routes for every subset), the day-2 routes and the week's costs.

```bash
python tests/test_dp_dry_run.py
# RESULT: dry run reproduced exactly
```

The core of it:

```python
inst = toy_instance()                       # Q=4, nV=2, TL=30, CCV=50, TU=5
sol = decode(inst, POP, MASK)               # greedy: A | B->C->D (32 min)

pts = np.array([0, 1, 2, 3])                # A, B, C, D collected on day 2
closed, dp, parent, end = held_karp_all_subsets(
    pts, inst.C, np.array([1.5, 1.5, 1.0, 1.5]), inst.tu)
assert closed[0b1110] == 29.0               # BCD, via C -> D -> B

ok, total, nr, ptr, stops = dp_day_routes(
    pts, np.array([3.0, 1.5, 0.8, 1.2]), sol.bins, inst.C,
    inst.bin_service, inst.Q, inst.TL, inst.n_vehicles, inst.tu)
assert total == 45.5                        # routes: A | C -> D -> B

pol = polish(inst, sol)
assert pol.overall_cost == 5105.0 and pol.feasible
```
