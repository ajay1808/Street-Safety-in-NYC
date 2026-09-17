# Street Safety in NYC

Ranking motor-vehicle crash risk at the **street-segment × hour** level in Manhattan.

Most road-safety analysis is retrospective — it maps where crashes *have* happened.
This project asks a forward-looking question instead: given a specific block of a
specific street, at 2pm on a Thursday, how risky is it relative to everywhere else?
Crashes are snapped onto individual road segments, so the unit of prediction is an
actual block of street rather than a zip code or a grid cell.

v1 rendered the result as a map. v2 rebuilds the modelling underneath it and reports
how well the ranking actually performs; the map is not yet rebuilt on top.

## Status

| Version | State | Notes |
|---|---|---|
| **v1** | Archived in [`v1/`](v1/) | Written August 2022. Runs, produces a map — but the headline metrics don't mean what they say. See below. |
| **v2** | Built — [`v2/`](v2/) | Rebuilt September 2026. Corrected sampling, rare-event metrics, 34 tests. [Results](v2/README.md#results). |

v1 is kept in the repository as-written, not quietly corrected. The gap between the
two versions is the most useful thing in here.

## Repository layout

```
.
├── README.md                # you are here
├── v1/                      # the 2022 original, archived as-is
│   ├── README.md            # what it does, how to run it, full list of known issues
│   ├── Motor_Vehicle_Safety_in_Manhattan_Project.ipynb
│   └── Borough Boundaries.geojson
└── v2/                      # the rebuild
    ├── README.md            # results, the prior correction, what changed
    ├── src/street_safety/   # data, panel, features, model, evaluate, pipeline
    └── tests/               # 34 tests
```

## v1: what it got right

Written in August 2022, before I started my master's in urban data science — this was
me learning the domain. The framing holds up better than the execution:

- **Segment-level, not area-level.** Crashes are snapped to OSM edges with
  `osmnx.distance.nearest_edges`, so predictions land on real road segments with real
  attributes (`highway` class, `maxspeed`, `oneway`, `length`).
- **Cyclic time features.** Month, hour, and weekday are treated as recurring
  structure rather than a linear timestamp.
- **A genuinely forward-looking output.** It scores every edge for all 24 hours of
  the next day and renders the result, rather than plotting history.

## v1: what it got wrong

The core problem is a sampling error that invalidates every number the notebook
reports.

The full panel is every edge × 12 months × 24 hours × 7 weekdays — roughly 19 million
rows, against 5,399 Manhattan crashes. The notebook keeps **all 5,399 crashes** but
randomly samples only **~10,000** of the 19 million non-crash rows. The training set
ends up at 15,395 rows with 5,399 positives:

|  | v1 training set | Reality |
|---|---|---|
| Crash rate | **35%** | **~0.03%** |

That's roughly a 1,000× oversample of the positive class, never corrected with class
weights or a prior adjustment. Everything downstream inherits it.

**The full list:**

1. **Sampling never corrected.** All positives kept, negatives subsampled ~1,000×,
   no reweighting. The model is fit to a world where a third of street-hours have a
   crash.
2. **Accuracy is the wrong metric.** At a 0.03% base rate, predicting "no crash"
   everywhere scores 99.97%. The reported 82% accuracy (0.68 precision / 0.86 recall
   on the positive class) describes only the resampled distribution.
3. **The output is not a probability.** `SVC(probability=False)` cannot emit one, and
   the commented-out `LinearRegression` on a 0/1 target gives unbounded values. The
   notebook nonetheless renders "Accident probability: 0.899" and builds a Safety
   Score and Star Rating on top of it.
4. **Imputation leak on `maxspeed`.** It is filled with `'25 mph'` *before* the road
   attributes are re-merged from the edge table, and never re-derived — so crash-side
   rows disproportionately carry the default. The model can separate classes partly on
   an imputation artifact.
5. **An outer merge defeats the sample.** `edges.merge(df_cal2, on='uvkey',
   how='outer')` re-adds every unsampled edge with null time fields, yielding 13,608
   rows instead of 10,000. They are dropped later, silently.
6. **`uvkey` is assigned wrong.** `crashes_manhattan['uvkey'] = key` stores the
   integer OSM key instead of the `"u v key"` composite string. Harmless in practice —
   the merge uses `u`/`v`/`key` separately — but wrong.
7. **Not reproducible.** Unseeded `.sample()`, `random_state=None` on the classifier,
   a `matplotlib==3.1.3` pin, hardcoded `/content/` Colab paths, and a blanket
   `warnings.filterwarnings('ignore')` masking chained-assignment warnings from
   `for i in range(len(df))` loops.
8. **Random split on time-series data.** A model whose whole purpose is predicting
   *tomorrow* is validated on a random split of an edge×time panel, drawn from a
   single thin window of the 30,000 most recent citywide crashes.

## v2: how each of these is addressed

Built and measured — full detail in [`v2/README.md`](v2/README.md).

| # | v1 problem | v2 approach |
|---|---|---|
| 1 | Case-control sampling left uncorrected | Sampling kept, but τ is computed exactly and the intercept corrected via King & Zeng (2001) |
| 2 | Accuracy at a 0.03% base rate | Accuracy is not computed at all. PR-AUC, precision@k and lift@k, against three baselines |
| 3 | Uncalibrated score labelled a probability | Ranking is the headline; a score earns the word "probability" only after correction, and calibration is measured |
| 4 | `maxspeed` imputation leak | NYC CSCL `posted_speed` is 91% populated; imputation moved inside a `Pipeline` fitted on training rows only |
| 5 | Outer merge inflating the frame | Negatives built explicitly by rejection sampling, with row-count assertions |
| 6 | Wrong `uvkey` construction | CSCL's `physicalid` is a single stable id — that bug class is gone by construction |
| 7 | Unreproducible environment | One seed, pinned requirements, `FutureWarning` promoted to an error |
| 8 | Random split, thin time window | Time-based split over 62,759 snapped crashes, 2021–2026 |

### Headline result

Evaluated on the **complete, unsampled** panel — 11,214 segments × 672 hours =
7,535,808 cells, 780 crash-hours, base rate 0.0104%:

| Model | Avg precision | ROC-AUC | lift@5000 |
|---|---|---|---|
| Gradient boosting | **0.000653** | **0.838** | **25.1×** |
| Baseline: prior crashes | 0.000518 | 0.820 | 19.3× |
| Baseline: random | 0.000111 | 0.517 | 1.9× |

The prior correction moves the logistic intercept from −2.377 to −9.125, which
brings mean predicted probability to within **1.6×** of the observed rate. v1's
numbers were off by about three orders of magnitude.

And the honest part: a no-model baseline of "this segment has had crashes
before" already reaches 19.3× lift. The model adds 26% on average precision
over that — real, but not transformative. Predicting the specific hour a crash
hits a specific block remains close to impossible; ranking relative risk is what
works.

## Data sources

- [NYC OpenData — Motor Vehicle Collisions](https://data.cityofnewyork.us/resource/h9gi-nx95) — both versions
- [NYC OpenData — CSCL Centerline](https://data.cityofnewyork.us/resource/inkn-q76z) — the street network in **v2**
- [OpenStreetMap](https://www.openstreetmap.org/) drive network via [OSMnx](https://osmnx.readthedocs.io/) — the street network in **v1**
- [NYC OpenData — Borough Boundaries](https://data.cityofnewyork.us/City-Government/Borough-Boundaries/tqmj-j8zm) — v1 only

v2 switched the network source from OpenStreetMap to the city's own centerline;
[the reasoning is in `v2/README.md`](v2/README.md#why-the-data-source-changed).
