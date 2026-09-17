# Street Safety in NYC

Predicting motor-vehicle accident risk at the **street-segment × hour** level in
Manhattan, and rendering it as a map you can read for any hour of an upcoming day.

Most road-safety analysis is retrospective — it maps where crashes *have* happened.
This project asks a forward-looking question instead: given a specific block of a
specific street, at 2pm on a Thursday, how risky is it relative to everywhere else?
Crashes are joined onto the OpenStreetMap drive network, so the unit of prediction is
an actual road segment rather than a zip code or a grid cell.

## Status

| Version | State | Notes |
|---|---|---|
| **v1** | Archived in [`v1/`](v1/) | Written August 2022. Runs, produces a map — but the headline metrics don't mean what they say. See below. |
| **v2** | Planned | Rebuild that fixes the sampling and metric problems described below. |

v1 is kept in the repository as-written, not quietly corrected. The gap between the
two versions is the most useful thing in here.

## Repository layout

```
.
├── README.md                # you are here
└── v1/                      # the 2022 original, archived as-is
    ├── README.md            # what it does, how to run it, full list of known issues
    ├── Motor_Vehicle_Safety_in_Manhattan_Project.ipynb
    └── Borough Boundaries.geojson
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

| # | v1 problem | v2 approach |
|---|---|---|
| 1 | Case-control sampling left uncorrected | Keep the sampling — it is a legitimate technique — but correct for it, via class weights or a King & Zeng rare-events intercept adjustment |
| 2 | Accuracy at a 0.03% base rate | Report **PR-AUC** and **precision@k** as headline metrics, against two baselines: predict-nothing, and crashes ∝ segment length |
| 3 | Uncalibrated score labelled a probability | Emit a **relative risk ranking**, which is what the map actually needs; call it a probability only once it is calibrated and reliability-checked |
| 4 | `maxspeed` imputation leak | Derive all road attributes from one source after the join; impute after the train/test split, never before |
| 5 | Outer merge inflating the frame | Construct the negative sample explicitly, with row-count assertions between steps |
| 6 | Wrong `uvkey` construction | One helper for the composite key, covered by a test |
| 7 | Unreproducible environment | Seeded throughout, pinned environment, vectorized pandas, warnings surfaced rather than silenced |
| 8 | Random split, thin time window | **Time-based split** — train on an earlier period, test on a later one — over a multi-year crash pull |

The honest summary: v1 had a good question and a broken answer. v2 keeps the question.

## Data sources

- [NYC OpenData — Motor Vehicle Collisions](https://data.cityofnewyork.us/resource/h9gi-nx95)
- [NYC OpenData — Borough Boundaries](https://data.cityofnewyork.us/City-Government/Borough-Boundaries/tqmj-j8zm)
- [OpenStreetMap](https://www.openstreetmap.org/) drive network via [OSMnx](https://osmnx.readthedocs.io/)
