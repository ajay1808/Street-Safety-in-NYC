"""Tests for feature construction, focused on leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from street_safety.features import SegmentHistory, add_time_features, build_design_frame

SEGMENTS_IDS = np.array([1, 2, 3], dtype=np.int64)
TRAIN_START = pd.Timestamp("2024-01-01")
TRAIN_END = pd.Timestamp("2024-02-01")
TEST_START = TRAIN_END


def make_segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "physicalid": SEGMENTS_IDS,
            "rw_type": ["1", "1", "2"],
            "trafdir": ["FT", "TW", "TF"],
            "snow_priority": ["C", "S", "V"],
            "posted_speed": [25.0, np.nan, 35.0],
            "segmentlength": [100.0, 200.0, 300.0],
            "streetwidth": [30.0, 40.0, 50.0],
            "number_travel_lanes": [2.0, 2.0, 4.0],
            "number_park_lanes": [1.0, 2.0, 0.0],
            "number_total_lanes": [3.0, 4.0, 4.0],
        }
    )


def test_history_excludes_everything_after_the_boundary():
    """The leak that would matter most: test-period crashes in a feature."""
    snapped = pd.DataFrame(
        {
            "physicalid": [1, 1, 2],
            "hour_ts": [
                TRAIN_START,  # in train window
                TEST_START + pd.Timedelta(hours=3),  # after the boundary
                TRAIN_START + pd.Timedelta(hours=10),  # in train window
            ],
        }
    )
    history = SegmentHistory.from_window(snapped, SEGMENTS_IDS, TRAIN_START, TRAIN_END)
    assert history.counts.loc[1] == 1  # not 2
    assert history.counts.loc[2] == 1
    assert history.counts.loc[3] == 0


def test_history_is_identical_across_splits():
    """The same history object is applied to train and test, unchanged."""
    snapped = pd.DataFrame({"physicalid": [1, 1, 2], "hour_ts": [TRAIN_START] * 3})
    history = SegmentHistory.from_window(snapped, SEGMENTS_IDS, TRAIN_START, TRAIN_END)

    train_cells = pd.DataFrame({"physicalid": [1], "hour_ts": [TRAIN_START]})
    test_cells = pd.DataFrame({"physicalid": [1], "hour_ts": [TEST_START]})

    assert (
        history.attach(train_cells)["hist_crash_count"].iloc[0]
        == history.attach(test_cells)["hist_crash_count"].iloc[0]
    )


def test_segments_with_no_history_get_zero_not_nan():
    snapped = pd.DataFrame({"physicalid": [1], "hour_ts": [TRAIN_START]})
    history = SegmentHistory.from_window(snapped, SEGMENTS_IDS, TRAIN_START, TRAIN_END)
    cells = pd.DataFrame({"physicalid": [3], "hour_ts": [TRAIN_START]})
    attached = history.attach(cells)
    assert attached["hist_crash_count"].iloc[0] == 0.0
    assert not attached["hist_crash_count"].isna().any()


def test_cyclic_hour_encoding_wraps():
    """Hour 23 must be adjacent to hour 0, not maximally distant."""
    frame = pd.DataFrame(
        {"hour_ts": pd.to_datetime(["2024-01-01 23:00", "2024-01-02 00:00", "2024-01-01 12:00"])}
    )
    out = add_time_features(frame)
    p23 = np.array([out["hour_sin"][0], out["hour_cos"][0]])
    p00 = np.array([out["hour_sin"][1], out["hour_cos"][1]])
    p12 = np.array([out["hour_sin"][2], out["hour_cos"][2]])
    assert np.linalg.norm(p23 - p00) < np.linalg.norm(p23 - p12)


def test_weekend_flag():
    frame = pd.DataFrame(
        {"hour_ts": pd.to_datetime(["2024-01-06 10:00", "2024-01-08 10:00"])}
    )  # Saturday, Monday
    out = add_time_features(frame)
    assert out["is_weekend"].tolist() == [1, 0]


def test_design_frame_join_preserves_row_count():
    """v1's merges silently changed row counts; this one asserts."""
    snapped = pd.DataFrame({"physicalid": [1], "hour_ts": [TRAIN_START]})
    history = SegmentHistory.from_window(snapped, SEGMENTS_IDS, TRAIN_START, TRAIN_END)
    cells = pd.DataFrame(
        {
            "physicalid": [1, 2, 3, 1],
            "hour_ts": pd.to_datetime(
                ["2024-02-01", "2024-02-01", "2024-02-01", "2024-02-02"]
            ),
            "target": [0, 0, 0, 1],
        }
    )
    out = build_design_frame(cells, make_segments(), history)
    assert len(out) == len(cells)


def test_missing_posted_speed_is_left_for_the_pipeline_to_impute():
    """Imputation must not happen before the split -- v1's maxspeed bug."""
    snapped = pd.DataFrame({"physicalid": [1], "hour_ts": [TRAIN_START]})
    history = SegmentHistory.from_window(snapped, SEGMENTS_IDS, TRAIN_START, TRAIN_END)
    cells = pd.DataFrame({"physicalid": [2], "hour_ts": [TRAIN_START], "target": [0]})
    out = build_design_frame(cells, make_segments(), history)
    assert out["posted_speed"].isna().all()
