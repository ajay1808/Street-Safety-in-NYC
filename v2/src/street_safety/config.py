"""Constants and configuration.

Every knob that affects results lives here, so a run is fully described by
this file plus the seed.
"""

from __future__ import annotations

from pathlib import Path

SEED = 20260917

# --- paths -----------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parent
V2_ROOT = PKG_ROOT.parent.parent
CACHE_DIR = V2_ROOT / "data" / "cache"
OUTPUT_DIR = V2_ROOT / "outputs"

# --- data sources ----------------------------------------------------------
SOCRATA_BASE = "https://data.cityofnewyork.us/resource"
CENTERLINE_DATASET = "inkn-q76z"  # NYC CSCL Centerline
CRASH_DATASET = "h9gi-nx95"  # NYC Motor Vehicle Collisions - Crashes

BOROUGH_CODE = "1"  # Manhattan, in the centerline schema
BOROUGH_NAME = "MANHATTAN"  # Manhattan, in the crash schema

# Drivable roadway types in the CSCL schema:
# 1 Street, 2 Highway, 3 Bridge, 4 Tunnel.
# Excludes ramps, ferry routes, alleys and non-roadway features.
DRIVABLE_RW_TYPES = ("1", "2", "3", "4")

# --- temporal windows ------------------------------------------------------
# Crashes are pulled from DATA_START onward. The model trains on cells before
# TRAIN_END and is evaluated on a full unsampled panel covering TEST_WEEKS
# weeks from TEST_START.
DATA_START = "2021-01-01"
TRAIN_END = "2026-01-01"
TEST_START = "2026-01-01"
TEST_WEEKS = 4

# --- sampling --------------------------------------------------------------
# Case-control sampling: keep every positive cell, draw
# NEGATIVES_PER_POSITIVE negatives per positive. This is corrected for at
# scoring time -- see model.prior_corrected_intercept.
NEGATIVES_PER_POSITIVE = 10

# --- evaluation ------------------------------------------------------------
PRECISION_AT_K = (100, 500, 1000, 5000)
SCORE_CHUNK_HOURS = 168  # score the test panel one week at a time

# --- feature groups --------------------------------------------------------
NUMERIC_SEGMENT_FEATURES = [
    "posted_speed",
    "segmentlength",
    "streetwidth",
    "number_travel_lanes",
    "number_park_lanes",
    "number_total_lanes",
    "hist_crash_count",
    "hist_crash_rate",
]
CATEGORICAL_SEGMENT_FEATURES = ["rw_type", "trafdir", "snow_priority"]
CYCLIC_TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]
BINARY_TIME_FEATURES = ["is_weekend"]

FEATURE_COLUMNS = (
    NUMERIC_SEGMENT_FEATURES
    + CATEGORICAL_SEGMENT_FEATURES
    + CYCLIC_TIME_FEATURES
    + BINARY_TIME_FEATURES
)
