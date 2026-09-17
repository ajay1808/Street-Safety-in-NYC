"""Feature construction.

Two rules, both of them things v1 broke:

1. **Nothing is imputed before the split.** v1 filled ``maxspeed`` with a
   constant while the crash rows and the sampled non-crash rows were still in
   different states of merge, so the positives disproportionately carried the
   fill value and the model could partly separate the classes on an artifact of
   imputation. Here, imputation lives inside a scikit-learn ``Pipeline`` and is
   fitted on training rows only.

2. **History is computed from the training window only.** ``hist_crash_count``
   is the strongest single predictor, and counting it over the full dataset
   would leak the test period into training. ``SegmentHistory`` is built from a
   window that ends at the train/test boundary and is then applied unchanged to
   both sides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


@dataclass(frozen=True)
class SegmentHistory:
    """Per-segment crash history measured over a fixed, past-only window."""

    counts: pd.Series  # physicalid -> crashes observed in the window
    window_hours: int

    @classmethod
    def from_window(
        cls,
        snapped: pd.DataFrame,
        segment_ids: np.ndarray,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> "SegmentHistory":
        from .panel import n_hours_between

        window = snapped[(snapped["hour_ts"] >= start) & (snapped["hour_ts"] < end)]
        counts = (
            window.groupby("physicalid").size().reindex(segment_ids, fill_value=0)
        )
        counts.index.name = "physicalid"
        return cls(counts=counts.astype("float64"), window_hours=n_hours_between(start, end))

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add ``hist_crash_count`` and ``hist_crash_rate`` to a cell frame."""
        out = frame.copy()
        count = self.counts.reindex(out["physicalid"]).to_numpy()
        out["hist_crash_count"] = np.nan_to_num(count, nan=0.0)
        out["hist_crash_rate"] = out["hist_crash_count"] / self.window_hours
        return out


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Cyclic encodings of hour, weekday and month.

    Sin/cos pairs keep hour 23 adjacent to hour 0 instead of maximally far
    apart, which a raw integer encoding would imply.
    """
    out = frame.copy()
    ts = out["hour_ts"]
    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    month = ts.dt.month.to_numpy()

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    out["is_weekend"] = (dow >= 5).astype(np.int8)
    return out


def build_design_frame(
    cells: pd.DataFrame,
    segments: pd.DataFrame,
    history: SegmentHistory,
) -> pd.DataFrame:
    """Join segment attributes and time features onto a frame of cells.

    Attributes come from exactly one source -- the centerline table -- so a
    column cannot arrive half-populated the way v1's ``maxspeed`` did.
    """
    attrs = segments[
        ["physicalid", *config.CATEGORICAL_SEGMENT_FEATURES]
        + [
            c
            for c in config.NUMERIC_SEGMENT_FEATURES
            if c not in ("hist_crash_count", "hist_crash_rate")
        ]
    ]
    n_before = len(cells)
    out = cells.merge(attrs, on="physicalid", how="left", validate="many_to_one")
    assert len(out) == n_before, "join changed row count"

    out = history.attach(out)
    out = add_time_features(out)
    return out


def make_preprocessor() -> ColumnTransformer:
    """Imputation, scaling and encoding -- all fitted on training rows only."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
        ]
    )
    passthrough_cols = config.CYCLIC_TIME_FEATURES + config.BINARY_TIME_FEATURES
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_SEGMENT_FEATURES),
            ("cat", categorical, config.CATEGORICAL_SEGMENT_FEATURES),
            ("time", "passthrough", passthrough_cols),
        ],
        remainder="drop",
    )
