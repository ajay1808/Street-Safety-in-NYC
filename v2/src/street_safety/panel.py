"""Build the (segment, hour) panel.

This module is where v1 went wrong, so the contract is spelled out.

A *cell* is one (segment, hour) pair. Its target is 1 if at least one crash was
reported on that segment during that hour. Over the training window there are
``n_segments x n_hours`` cells -- on the order of half a billion -- against
roughly sixty thousand positives, so the positive rate is about 0.01%.

Enumerating that panel is not practical, and training on it directly would be
wasteful, so we use **case-control sampling**: keep every positive, draw a
modest multiple of negatives. That is a legitimate and standard technique. What
v1 did wrong was not the sampling but the failure to correct for it: it fitted
on a 35% positive rate and then reported the resulting scores as if they were
real-world probabilities.

Two things here keep that from happening again:

1. ``TrainingSample.tau`` records the *true* population positive rate, computed
   exactly rather than estimated. ``model`` uses it to correct the fitted
   intercept back onto the population scale.
2. Evaluation never touches a sampled frame. ``iter_test_panel_chunks`` yields
   the **complete, unsampled** panel for the test window, so reported metrics
   are measured at the real base rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from . import config

HOUR = pd.Timedelta(hours=1)


def hour_index(start: pd.Timestamp, n_hours: int) -> pd.DatetimeIndex:
    """The hourly timestamps of a window, left-closed and right-open."""
    return pd.date_range(start=start, periods=n_hours, freq="h")


def n_hours_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Whole hours in ``[start, end)``."""
    delta = end - start
    if delta <= pd.Timedelta(0):
        raise ValueError(f"end ({end}) must be after start ({start})")
    return int(delta // HOUR)


def positive_cells(
    snapped: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Distinct (segment, hour) cells with at least one crash in the window.

    Multiple crashes in the same cell collapse to one positive: the target is
    "did a crash happen here", not "how many".
    """
    window = snapped[(snapped["hour_ts"] >= start) & (snapped["hour_ts"] < end)]
    cells = (
        window[["physicalid", "hour_ts"]]
        .drop_duplicates()
        .sort_values(["hour_ts", "physicalid"])
        .reset_index(drop=True)
    )
    return cells


@dataclass(frozen=True)
class TrainingSample:
    """A case-control sample plus everything needed to undo the sampling.

    Attributes
    ----------
    frame:
        Columns ``physicalid``, ``hour_ts``, ``target``.
    tau:
        True positive rate in the population the sample was drawn from.
    sample_rate:
        Positive rate within ``frame``. The gap between this and ``tau`` is
        exactly what the intercept correction removes.
    n_population_cells:
        Size of the full (unsampled) panel for the window.
    """

    frame: pd.DataFrame
    tau: float
    sample_rate: float
    n_population_cells: int
    n_positives: int
    n_negatives: int

    def describe(self) -> str:
        return (
            f"population cells   {self.n_population_cells:,}\n"
            f"positives          {self.n_positives:,}\n"
            f"negatives sampled  {self.n_negatives:,}\n"
            f"tau (true rate)    {self.tau:.6%}\n"
            f"sampled rate       {self.sample_rate:.4%}\n"
            f"oversampling       {self.sample_rate / self.tau:,.0f}x"
        )


def build_training_sample(
    snapped: pd.DataFrame,
    segment_ids: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    negatives_per_positive: int = config.NEGATIVES_PER_POSITIVE,
    seed: int = config.SEED,
) -> TrainingSample:
    """Every positive cell in the window, plus sampled negatives.

    Negatives are drawn uniformly from the full cell space by rejection
    sampling: a draw that lands on a positive, or duplicates an earlier draw,
    is discarded. That keeps the negatives an unbiased sample of non-crash
    cells, which is what makes the intercept correction valid.
    """
    rng = np.random.default_rng(seed)

    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    n_segments = segment_ids.size
    n_hours = n_hours_between(start, end)
    n_population_cells = n_segments * n_hours

    pos = positive_cells(snapped, start, end)
    pos = pos[pos["physicalid"].isin(segment_ids)].reset_index(drop=True)
    n_pos = len(pos)
    if n_pos == 0:
        raise ValueError("no positive cells in the requested window")

    # Dense index so a cell fits in one int64 we can hash cheaply.
    seg_pos = pd.Series(np.arange(n_segments, dtype=np.int64), index=segment_ids)
    pos_seg_idx = seg_pos.loc[pos["physicalid"]].to_numpy()
    pos_hour_idx = ((pos["hour_ts"] - start) // HOUR).to_numpy().astype(np.int64)
    pos_codes = pos_seg_idx * n_hours + pos_hour_idx
    pos_lookup = np.sort(pos_codes)

    target_negatives = n_pos * negatives_per_positive
    if target_negatives > n_population_cells - n_pos:
        raise ValueError("asked for more negatives than the panel contains")

    collected: list[np.ndarray] = []
    n_collected = 0
    # Collisions are rare (positives are ~0.01% of cells), so a small
    # over-draw converges in one or two rounds.
    while n_collected < target_negatives:
        draw_size = int((target_negatives - n_collected) * 1.1) + 1024
        draws = rng.integers(0, n_population_cells, size=draw_size, dtype=np.int64)
        idx = np.searchsorted(pos_lookup, draws)
        idx = np.clip(idx, 0, pos_lookup.size - 1)
        is_positive = pos_lookup[idx] == draws
        draws = draws[~is_positive]
        collected.append(draws)
        n_collected = np.unique(np.concatenate(collected)).size

    neg_codes = np.unique(np.concatenate(collected))
    # unique() may overshoot; trim deterministically.
    if neg_codes.size > target_negatives:
        neg_codes = rng.permutation(neg_codes)[:target_negatives]
        neg_codes.sort()

    neg_seg_idx, neg_hour_idx = np.divmod(neg_codes, n_hours)
    negatives = pd.DataFrame(
        {
            "physicalid": segment_ids[neg_seg_idx],
            "hour_ts": start + neg_hour_idx * HOUR,
            "target": np.zeros(neg_codes.size, dtype=np.int8),
        }
    )
    positives = pd.DataFrame(
        {
            "physicalid": pos["physicalid"].to_numpy(),
            "hour_ts": pos["hour_ts"].to_numpy(),
            "target": np.ones(n_pos, dtype=np.int8),
        }
    )

    frame = (
        pd.concat([positives, negatives], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    # Guard rails. v1 lost 3,608 rows to an outer merge without noticing.
    assert len(frame) == n_pos + neg_codes.size, "row count drifted during concat"
    assert frame["target"].sum() == n_pos, "positive count drifted"
    assert not frame.duplicated(subset=["physicalid", "hour_ts"]).any(), (
        "duplicate cells in training sample"
    )

    tau = n_pos / n_population_cells
    sample_rate = n_pos / len(frame)
    return TrainingSample(
        frame=frame,
        tau=tau,
        sample_rate=sample_rate,
        n_population_cells=n_population_cells,
        n_positives=n_pos,
        n_negatives=int(neg_codes.size),
    )


def iter_test_panel_chunks(
    snapped: pd.DataFrame,
    segment_ids: np.ndarray,
    start: pd.Timestamp,
    n_hours: int,
    *,
    chunk_hours: int = config.SCORE_CHUNK_HOURS,
) -> Iterator[pd.DataFrame]:
    """Yield the complete, unsampled test panel in time-ordered chunks.

    No sampling happens here. Every segment appears for every hour, so metrics
    computed over these chunks sit at the real base rate.
    """
    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    end = start + n_hours * HOUR
    pos = positive_cells(snapped, start, end)
    pos_key = set(zip(pos["physicalid"].to_numpy(), pos["hour_ts"].to_numpy()))

    for chunk_start_hour in range(0, n_hours, chunk_hours):
        size = min(chunk_hours, n_hours - chunk_start_hour)
        hours = hour_index(start + chunk_start_hour * HOUR, size)

        frame = pd.DataFrame(
            {
                "physicalid": np.repeat(segment_ids, size),
                "hour_ts": np.tile(hours.to_numpy(), segment_ids.size),
            }
        )
        keys = list(zip(frame["physicalid"].to_numpy(), frame["hour_ts"].to_numpy()))
        frame["target"] = np.fromiter(
            (k in pos_key for k in keys), dtype=np.int8, count=len(keys)
        )
        assert len(frame) == segment_ids.size * size, "test chunk is not complete"
        yield frame
