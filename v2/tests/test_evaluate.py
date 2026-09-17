"""Tests for the metrics, on hand-checkable arrays."""

from __future__ import annotations

import numpy as np
import pytest

from street_safety.evaluate import (
    ScoredPanel,
    calibration_table,
    evaluate_scores,
    precision_recall_at_k,
)


def test_precision_at_k_picks_the_top_scores():
    y_true = np.array([0, 1, 0, 1, 0])
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    precision, recall = precision_recall_at_k(y_true, y_score, k=2)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_precision_at_k_with_no_hits():
    y_true = np.array([1, 0, 0, 0])
    y_score = np.array([0.1, 0.9, 0.8, 0.7])
    precision, recall = precision_recall_at_k(y_true, y_score, k=2)
    assert precision == 0.0
    assert recall == 0.0


def test_precision_at_k_clamps_k_to_array_length():
    y_true = np.array([1, 0])
    y_score = np.array([0.9, 0.1])
    precision, _ = precision_recall_at_k(y_true, y_score, k=100)
    assert precision == pytest.approx(0.5)


def test_accuracy_is_not_reported():
    """Deliberate: at a 0.01% base rate it is a meaningless number.

    v1's headline metric. Its absence here is the point.
    """
    y_true = np.zeros(1000, dtype=np.int8)
    y_true[:2] = 1
    row = evaluate_scores(y_true, np.random.default_rng(0).random(1000), name="m", ks=(10,))
    assert not any("accuracy" in key for key in row)
    assert "average_precision" in row
    assert "precision@10" in row


def test_lift_is_relative_to_the_base_rate():
    y_true = np.zeros(1000, dtype=np.int8)
    y_true[:10] = 1  # base rate 1%
    y_score = np.zeros(1000)
    y_score[:10] = 1.0  # perfect ranking
    row = evaluate_scores(y_true, y_score, name="perfect", ks=(10,))
    assert row["precision@10"] == pytest.approx(1.0)
    assert row["lift@10"] == pytest.approx(100.0)


def test_scored_panel_concatenates_chunks_in_order():
    panel = ScoredPanel()
    panel.add(np.array([0, 1]), np.array([0.1, 0.2]))
    panel.add(np.array([1, 0]), np.array([0.3, 0.4]))
    y_true, y_score = panel.finalize()
    np.testing.assert_array_equal(y_true, [0, 1, 1, 0])
    np.testing.assert_allclose(y_score, [0.1, 0.2, 0.3, 0.4])


def test_calibration_table_detects_a_well_calibrated_model():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 0.2, size=50_000)
    y = rng.binomial(1, p)
    table = calibration_table(y, p, n_bins=5)
    assert len(table) == 5
    # A calibrated model tracks the diagonal within sampling noise.
    assert np.allclose(table["mean_predicted"], table["observed_rate"], atol=0.02)


def test_calibration_table_detects_an_overconfident_model():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.0, 0.1, size=50_000)
    y = rng.binomial(1, p_true)
    p_claimed = p_true * 10  # systematically inflated, as v1's scores were
    table = calibration_table(y, p_claimed, n_bins=5)
    assert (table["mean_predicted"] > table["observed_rate"] * 2).all()
