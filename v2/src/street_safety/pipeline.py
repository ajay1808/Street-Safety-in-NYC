"""End-to-end run: fetch, build, fit, evaluate, write results."""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from . import config, data, evaluate, model
from .features import SegmentHistory, build_design_frame
from .panel import build_training_sample, iter_test_panel_chunks

log = logging.getLogger(__name__)


def run(*, force_refresh: bool = False) -> pd.DataFrame:
    started = time.time()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_start = pd.Timestamp(config.DATA_START)
    train_end = pd.Timestamp(config.TRAIN_END)
    test_start = pd.Timestamp(config.TEST_START)
    test_hours = config.TEST_WEEKS * 168

    log.info("loading source data")
    segments = data.load_centerline(force=force_refresh)
    crashes = data.load_crashes(force=force_refresh)
    snapped = data.snap_crashes_to_segments(crashes, segments)
    segment_ids = segments["physicalid"].to_numpy()
    log.info("%d segments, %d snapped crashes", len(segments), len(snapped))

    # History is measured on the training window only, then applied unchanged
    # to both splits -- see features.SegmentHistory.
    history = SegmentHistory.from_window(snapped, segment_ids, train_start, train_end)

    log.info("building training sample")
    sample = build_training_sample(
        snapped, segment_ids, train_start, train_end,
        negatives_per_positive=config.NEGATIVES_PER_POSITIVE,
        seed=config.SEED,
    )
    log.info("\n%s", sample.describe())

    train_df = build_design_frame(sample.frame, segments, history)
    X_train = train_df[config.FEATURE_COLUMNS]
    y_train = train_df["target"].to_numpy()

    log.info("fitting models")
    fitted = [
        model.fit_logistic(X_train, y_train, tau=sample.tau, sample_rate=sample.sample_rate),
        model.fit_gradient_boosting(X_train, y_train, tau=sample.tau, sample_rate=sample.sample_rate),
    ]
    for m in fitted:
        log.info("  fitted %s", m.name)

    log.info(
        "scoring full unsampled test panel: %d segments x %d hours = %s cells",
        len(segment_ids), test_hours, f"{len(segment_ids) * test_hours:,}",
    )
    scored = {m.name: evaluate.ScoredPanel() for m in fitted}
    scored.update({b: evaluate.ScoredPanel() for b in evaluate.BASELINES})
    prob_panel = evaluate.ScoredPanel()

    n_chunks = 0
    for chunk in iter_test_panel_chunks(snapped, segment_ids, test_start, test_hours):
        design = build_design_frame(chunk, segments, history)
        X = design[config.FEATURE_COLUMNS]
        y = design["target"].to_numpy()

        for m in fitted:
            scored[m.name].add(y, m.rank_scores(X))
            if m.corrected_intercept is not None:
                probs = m.population_probabilities(X)
                if probs is not None and m.name == "logistic_regression":
                    prob_panel.add(y, probs)
        for bname, fn in evaluate.BASELINES.items():
            scored[bname].add(y, fn(design))

        n_chunks += 1
        log.info("  chunk %d: %d cells, %d positives", n_chunks, len(design), int(y.sum()))

    rows = []
    for name, panel in scored.items():
        y_true, y_score = panel.finalize()
        rows.append(evaluate.evaluate_scores(y_true, y_score, name=name))
    results = pd.DataFrame(rows).sort_values("average_precision", ascending=False)

    # Calibration of the prior-corrected logistic probabilities.
    y_true_p, y_prob = prob_panel.finalize()
    calib = evaluate.calibration_table(y_true_p, y_prob)

    results.to_csv(config.OUTPUT_DIR / "metrics.csv", index=False)
    calib.to_csv(config.OUTPUT_DIR / "calibration.csv", index=False)
    run_meta = {
        "seed": config.SEED,
        "data_start": config.DATA_START,
        "train_end": config.TRAIN_END,
        "test_start": config.TEST_START,
        "test_weeks": config.TEST_WEEKS,
        "negatives_per_positive": config.NEGATIVES_PER_POSITIVE,
        "n_segments": int(len(segment_ids)),
        "n_snapped_crashes": int(len(snapped)),
        "train_population_cells": int(sample.n_population_cells),
        "train_positives": int(sample.n_positives),
        "tau": sample.tau,
        "sample_positive_rate": sample.sample_rate,
        "oversampling_factor": sample.sample_rate / sample.tau,
        "logistic_raw_intercept": float(
            fitted[0].pipeline.named_steps["clf"].intercept_[0]
        ),
        "logistic_corrected_intercept": fitted[0].corrected_intercept,
        "mean_corrected_probability": float(np.mean(y_prob)),
        "test_observed_rate": float(np.mean(y_true_p)),
        "runtime_seconds": round(time.time() - started, 1),
    }
    (config.OUTPUT_DIR / "run.json").write_text(json.dumps(run_meta, indent=2))

    log.info("\n%s", results.to_string(index=False))
    log.info("\ncalibration\n%s", calib.to_string(index=False))
    log.info("\nwrote outputs to %s", config.OUTPUT_DIR)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
