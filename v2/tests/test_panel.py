"""Tests for panel construction.

Each test here corresponds to a specific way v1 went wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from street_safety.panel import (
    build_training_sample,
    iter_test_panel_chunks,
    n_hours_between,
    positive_cells,
)

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2024-02-01")  # 31 days = 744 hours
SEGMENTS = np.arange(1, 51, dtype=np.int64)  # 50 segments


def make_snapped(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = rng.integers(0, n_hours_between(START, END), size=n)
    return pd.DataFrame(
        {
            "collision_id": np.arange(n),
            "physicalid": rng.choice(SEGMENTS, size=n),
            "hour_ts": START + pd.to_timedelta(hours, unit="h"),
            "snap_distance_ft": 1.0,
        }
    )


def test_n_hours_between_is_exact():
    assert n_hours_between(START, END) == 744
    with pytest.raises(ValueError):
        n_hours_between(END, START)


def test_repeat_crashes_in_one_cell_collapse_to_one_positive():
    """The target is 'did a crash happen', not 'how many'."""
    snapped = pd.DataFrame(
        {
            "collision_id": [1, 2, 3],
            "physicalid": [7, 7, 7],
            "hour_ts": [START, START, START + pd.Timedelta(hours=5)],
        }
    )
    cells = positive_cells(snapped, START, END)
    assert len(cells) == 2


def test_positive_cells_respects_window_boundaries():
    snapped = pd.DataFrame(
        {
            "collision_id": [1, 2, 3],
            "physicalid": [1, 2, 3],
            "hour_ts": [
                START - pd.Timedelta(hours=1),  # before
                START,  # inside (left-closed)
                END,  # on the boundary (right-open)
            ],
        }
    )
    cells = positive_cells(snapped, START, END)
    assert cells["physicalid"].tolist() == [2]


def test_training_sample_row_counts_are_exact():
    """v1 silently produced 13,608 rows where it intended 10,000."""
    snapped = make_snapped()
    sample = build_training_sample(
        snapped, SEGMENTS, START, END, negatives_per_positive=5, seed=1
    )
    assert sample.n_negatives == sample.n_positives * 5
    assert len(sample.frame) == sample.n_positives + sample.n_negatives
    assert int(sample.frame["target"].sum()) == sample.n_positives


def test_sampled_negatives_never_collide_with_positives():
    """A 'negative' that is actually a crash cell would corrupt the target."""
    snapped = make_snapped(n=400, seed=3)
    sample = build_training_sample(
        snapped, SEGMENTS, START, END, negatives_per_positive=8, seed=2
    )
    pos_keys = set(
        zip(*positive_cells(snapped, START, END)[["physicalid", "hour_ts"]].to_numpy().T)
    )
    negatives = sample.frame[sample.frame["target"] == 0]
    neg_keys = set(zip(negatives["physicalid"], negatives["hour_ts"]))
    assert pos_keys.isdisjoint(neg_keys)


def test_training_sample_has_no_duplicate_cells():
    snapped = make_snapped(n=300, seed=4)
    sample = build_training_sample(
        snapped, SEGMENTS, START, END, negatives_per_positive=6, seed=5
    )
    assert not sample.frame.duplicated(subset=["physicalid", "hour_ts"]).any()


def test_tau_is_the_true_population_rate_not_the_sample_rate():
    """The distinction v1 missed entirely."""
    snapped = make_snapped()
    sample = build_training_sample(
        snapped, SEGMENTS, START, END, negatives_per_positive=9, seed=6
    )
    expected_cells = SEGMENTS.size * n_hours_between(START, END)

    assert sample.n_population_cells == expected_cells
    assert sample.tau == pytest.approx(sample.n_positives / expected_cells)
    assert sample.sample_rate == pytest.approx(0.1, abs=0.01)
    assert sample.sample_rate > sample.tau * 10


def test_sampling_is_reproducible_under_a_fixed_seed():
    """v1's .sample(10000) had no random_state."""
    snapped = make_snapped()
    a = build_training_sample(snapped, SEGMENTS, START, END, seed=42)
    b = build_training_sample(snapped, SEGMENTS, START, END, seed=42)
    pd.testing.assert_frame_equal(a.frame, b.frame)


def test_test_panel_is_complete_and_unsampled():
    """Evaluation must see every cell, or the base rate is wrong."""
    snapped = make_snapped()
    n_hours = 48
    chunks = list(
        iter_test_panel_chunks(snapped, SEGMENTS, START, n_hours, chunk_hours=24)
    )
    assert len(chunks) == 2

    combined = pd.concat(chunks, ignore_index=True)
    assert len(combined) == SEGMENTS.size * n_hours
    assert not combined.duplicated(subset=["physicalid", "hour_ts"]).any()

    expected_positives = len(positive_cells(snapped, START, START + pd.Timedelta(hours=n_hours)))
    assert int(combined["target"].sum()) == expected_positives
