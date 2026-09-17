# v2 — Street-segment crash risk, rebuilt

Ranks Manhattan street segments by crash risk for a given hour, and reports how
well that ranking actually works at the real base rate.

v1 asked a good question and answered it with a broken measurement. v2 keeps the
question. The headline difference: **v1 trained on a 35% crash rate and reported
82% accuracy; the real rate is 0.0117%, and at that rate accuracy is meaningless.**

See the [v1 post-mortem](../v1/README.md) for what specifically went wrong.

## Results

Trained on 2021-01-01 → 2026-01-01. Evaluated on the **complete, unsampled**
panel for the following 4 weeks: 11,214 segments × 672 hours = **7,535,808
cells** containing **780 crash-hours** (base rate 0.0104%).

| Model | Avg precision | ROC-AUC | precision@5000 | recall@5000 | lift@5000 |
|---|---|---|---|---|---|
| Gradient boosting | **0.000653** | **0.838** | 0.26% | 1.67% | **25.1×** |
| Logistic regression | 0.000580 | 0.823 | 0.22% | 1.41% | 21.3× |
| Baseline: prior crashes | 0.000518 | 0.820 | 0.20% | 1.28% | 19.3× |
| Baseline: segment length | 0.000138 | 0.581 | 0.00% | 0.00% | 0.0× |
| Baseline: random | 0.000111 | 0.517 | 0.02% | 0.13% | 1.9× |

**Read this honestly.** The model works, but the margin that matters is
modest. Ranking the top 5,000 street-hours out of 7.5 million finds crashes at
25× the random rate — but "prior crashes on this segment", a baseline requiring
no model at all, already gets 19×. Gradient boosting improves average precision
over that baseline by 26%. That is a real gain, not a transformative one.

Absolute precision stays tiny (0.26%) because the event is genuinely rare.
Predicting *which specific hour* a crash hits *a specific block* is close to
impossible. Ranking relative risk is tractable. Those are different claims, and
v1 conflated them.

## The correction, in one number

Case-control sampling is legitimate; failing to correct for it is not. Keeping
every positive and 10 negatives per positive gives a training set with a 9.09%
positive rate against a population rate of 0.0117% — a **775× oversample**,
almost exactly the distortion v1 shipped uncorrected.

For logistic regression the distortion lands entirely in the intercept, and
[King & Zeng (2001)](https://gking.harvard.edu/files/0s.pdf) give the fix:

```
β₀_corrected = β₀ − ln[ ((1−τ)/τ) × (ȳ/(1−ȳ)) ]
```

| | Value |
|---|---|
| Fitted intercept (sample scale) | −2.377 |
| Corrected intercept (population scale) | **−9.125** |
| Mean predicted probability, corrected | 0.000166 |
| Observed rate in test period | 0.000104 |

The corrected predictions average within **1.6×** of the observed rate. v1's
were off by roughly three orders of magnitude.

Slopes are untouched by the correction, so **the ranking is identical before and
after** — which is why v1's map retained some value even though its numbers did
not. `tests/test_model.py` proves both properties on synthetic data with a known
population rate.

### Calibration

Predicted vs. observed, by decile of predicted risk:

| Mean predicted | Observed | Ratio |
|---|---|---|
| 0.000003 | 0.000003 | 1.21 |
| 0.000012 | 0.000004 | 2.95 |
| 0.000027 | 0.000012 | 2.25 |
| 0.000039 | 0.000028 | 1.38 |
| 0.000051 | 0.000044 | 1.16 |
| 0.000065 | 0.000050 | 1.28 |
| 0.000081 | 0.000081 | 1.00 |
| 0.000104 | 0.000119 | 0.87 |
| 0.000150 | 0.000204 | 0.74 |
| 0.001132 | 0.000490 | 2.31 |

Seven of ten deciles land within 1.4× of observed. The riskiest decile
over-predicts by 2.3× — the model is overconfident exactly where it matters
most, which is the clearest open problem in this build.

## What changed from v1

| # | v1 problem | v2 |
|---|---|---|
| 1 | Case-control sampling, uncorrected | τ computed exactly (`panel.TrainingSample.tau`), intercept corrected via King & Zeng |
| 2 | Accuracy at a 0.01% base rate | Accuracy is not computed. PR-AUC, precision@k, lift@k, against three baselines |
| 3 | Uncalibrated score labelled "probability" | Ranking is the headline; a score is called a probability only after correction, and calibration is measured |
| 4 | `maxspeed` imputation leak | NYC CSCL `posted_speed` is 91% populated; all imputation happens inside a `Pipeline` fitted on train rows only |
| 5 | Outer merge silently added 3,608 rows | Negatives built explicitly by rejection sampling; row counts asserted |
| 6 | `uvkey` assigned the wrong value | CSCL's `physicalid` is a single stable segment id — the composite-key bug class is gone by construction |
| 7 | Unseeded sampling, 2020 pins, `filterwarnings('ignore')` | One seed in `config.SEED`; `pytest.ini` promotes `FutureWarning` to an error |
| 8 | Random split, 5,399 crashes from one thin window | Time-based split over **62,759** snapped crashes, 2021–2026 |

### Why the data source changed

v1 pulled the drive network from OpenStreetMap via OSMnx. v2 uses NYC's own
**CSCL Centerline**, which is authoritative for the city and better suited here:
a stable `physicalid` per segment (no `(u, v, key)` triple to mis-assemble), and
a populated `posted_speed` — removing the sparse field that caused v1's
imputation leak at the source. It also carries lane counts, street width,
traffic direction and snow priority, none of which OSM reliably has.

Crashes are snapped to segments with `geopandas.sjoin_nearest` in EPSG:2263,
capped at 100 ft. 62,759 of 63,175 crashes (99.3%) match, median distance 1.8 ft.

## Layout

```
v2/
├── src/street_safety/
│   ├── config.py      # every knob that affects results
│   ├── data.py        # Socrata fetch + caching, crash→segment snapping
│   ├── panel.py       # cell construction, case-control sampling, τ
│   ├── features.py    # leak-free features, past-only history
│   ├── model.py       # models + the King & Zeng correction
│   ├── evaluate.py    # PR-AUC, precision@k, calibration, baselines
│   └── pipeline.py    # end-to-end run
├── tests/             # 34 tests, offline (no network)
└── outputs/           # metrics.csv, calibration.csv, run.json
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python -m street_safety.pipeline   # ~40s, downloads on first run
PYTHONPATH=src pytest                              # 34 tests, no network
```

Source data is cached to `data/cache/` after the first fetch. The full run takes
about 40 seconds on a laptop, including scoring all 7.5M test cells in weekly
chunks.

## Known limitations

- **Overconfident at the top.** The riskiest decile over-predicts by 2.3×.
  Isotonic regression on a held-out slice is the obvious next step.
- **Gains over the historical baseline are modest.** Most of the signal is
  "this segment has had crashes before". The time features earn less than the
  framing implies.
- **No exposure data.** Crash risk should scale with traffic volume, which is
  absent here. Segment length is a poor proxy and the baseline results show it.
- **Reported crashes only.** The target is police-reported collisions, which
  under-count minor incidents and may under-report unevenly across neighborhoods.
- **Manhattan only**, and no weather, construction, or event data.
