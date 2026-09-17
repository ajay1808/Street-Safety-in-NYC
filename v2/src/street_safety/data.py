"""Fetch and cache the two source datasets, and snap crashes onto segments.

Sources
-------
NYC CSCL Centerline  -- the city's authoritative street centerline. Used instead
of the OpenStreetMap drive network that v1 pulled through OSMnx: it carries a
stable per-segment identifier (``physicalid``) and, critically, a populated
``posted_speed`` column. v1's ``maxspeed`` was sparse in OSM and had to be
imputed, which is where its imputation leak came from.

NYC Motor Vehicle Collisions -- one row per reported crash, with lat/lon.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape

from . import config

log = logging.getLogger(__name__)

# NY State Plane Long Island (ft) -- the standard projected CRS for NYC work.
# Distances below are in feet.
PROJECTED_CRS = "EPSG:2263"
WGS84 = "EPSG:4326"

# A crash further than this from any drivable centerline is dropped rather than
# force-matched to a distant segment.
MAX_SNAP_DISTANCE_FT = 100.0

_PAGE_SIZE = 25_000


def _socrata_fetch(dataset: str, params: dict[str, str]) -> list[dict]:
    """Page through a Socrata dataset until it stops returning rows."""
    url = f"{config.SOCRATA_BASE}/{dataset}.json"
    rows: list[dict] = []
    offset = 0
    while True:
        page_params = dict(params, **{"$limit": str(_PAGE_SIZE), "$offset": str(offset)})
        resp = requests.get(url, params=page_params, timeout=180)
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        rows.extend(page)
        log.info("%s: fetched %d rows (total %d)", dataset, len(page), len(rows))
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


def load_centerline(*, force: bool = False) -> gpd.GeoDataFrame:
    """Manhattan drivable street segments, one row per ``physicalid``."""
    cache = config.CACHE_DIR / "centerline.parquet"
    if cache.exists() and not force:
        return gpd.read_parquet(cache)

    rw_list = ",".join(f"'{t}'" for t in config.DRIVABLE_RW_TYPES)
    rows = _socrata_fetch(
        config.CENTERLINE_DATASET,
        {
            "$where": f"boroughcode='{config.BOROUGH_CODE}' AND rw_type in({rw_list})",
            "$order": "physicalid",
        },
    )
    df = pd.DataFrame(rows)

    geometry = [shape(g) for g in df["the_geom"]]
    gdf = gpd.GeoDataFrame(df.drop(columns=["the_geom"]), geometry=geometry, crs=WGS84)

    numeric = [
        "posted_speed",
        "segmentlength",
        "streetwidth",
        "number_travel_lanes",
        "number_park_lanes",
        "number_total_lanes",
    ]
    for col in numeric:
        gdf[col] = pd.to_numeric(gdf.get(col), errors="coerce")

    for col in ("rw_type", "trafdir", "snow_priority", "full_street_name"):
        gdf[col] = gdf.get(col).astype("string").fillna("UNKNOWN")

    gdf["physicalid"] = gdf["physicalid"].astype("int64")

    # A physicalid can appear more than once when a street is split across
    # administrative rows. Keep the longest piece so the id stays a key.
    gdf = (
        gdf.sort_values("segmentlength", ascending=False)
        .drop_duplicates(subset="physicalid", keep="first")
        .sort_values("physicalid")
        .reset_index(drop=True)
    )

    keep = [
        "physicalid",
        "full_street_name",
        "rw_type",
        "trafdir",
        "snow_priority",
        *numeric,
        "geometry",
    ]
    gdf = gdf[keep]

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(cache)
    return gdf


def load_crashes(*, force: bool = False) -> gpd.GeoDataFrame:
    """Manhattan crashes from ``config.DATA_START`` onward, with a timestamp."""
    cache = config.CACHE_DIR / "crashes.parquet"
    if cache.exists() and not force:
        return gpd.read_parquet(cache)

    rows = _socrata_fetch(
        config.CRASH_DATASET,
        {
            "$select": "collision_id,crash_date,crash_time,latitude,longitude",
            "$where": (
                f"borough='{config.BOROUGH_NAME}' "
                f"AND crash_date >= '{config.DATA_START}T00:00:00' "
                "AND latitude IS NOT NULL AND longitude IS NOT NULL "
                "AND latitude != 0"
            ),
            "$order": "crash_date",
        },
    )
    df = pd.DataFrame(rows)

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    # crash_date carries a date at midnight; crash_time carries HH:MM.
    date_part = df["crash_date"].str.slice(0, 10)
    ts = pd.to_datetime(
        date_part + " " + df["crash_time"].astype(str), format="mixed", errors="coerce"
    )
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"])

    # Floor to the hour: the modelling cell is (segment, hour).
    df["hour_ts"] = df["timestamp"].dt.floor("h")

    gdf = gpd.GeoDataFrame(
        df[["collision_id", "timestamp", "hour_ts"]],
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=WGS84,
    )

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(cache)
    return gdf


def snap_crashes_to_segments(
    crashes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    *,
    max_distance_ft: float = MAX_SNAP_DISTANCE_FT,
) -> pd.DataFrame:
    """Attach the nearest ``physicalid`` to each crash.

    Crashes with no drivable segment within ``max_distance_ft`` are dropped
    instead of being matched to whatever happens to be closest.
    """
    crashes_proj = crashes.to_crs(PROJECTED_CRS)
    segments_proj = segments[["physicalid", "geometry"]].to_crs(PROJECTED_CRS)

    joined = gpd.sjoin_nearest(
        crashes_proj,
        segments_proj,
        how="inner",
        max_distance=max_distance_ft,
        distance_col="snap_distance_ft",
    )

    # sjoin_nearest emits ties; keep the single closest match per crash.
    joined = (
        joined.sort_values("snap_distance_ft")
        .drop_duplicates(subset="collision_id", keep="first")
        .reset_index(drop=True)
    )

    out = pd.DataFrame(
        {
            "collision_id": joined["collision_id"],
            "physicalid": joined["physicalid"].astype("int64"),
            "hour_ts": joined["hour_ts"],
            "snap_distance_ft": joined["snap_distance_ft"],
        }
    )
    return out.sort_values("hour_ts").reset_index(drop=True)
