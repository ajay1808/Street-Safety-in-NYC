"""Metrics appropriate to a 0.01% base rate.

v1 reported accuracy. At the real base rate, predicting "no crash" everywhere
scores 99.99% accuracy, so the number carries no information at all. It is not
computed here.

What is computed:

``average_precision``
    Area under the precision-recall curve. Unlike ROC-AUC it does not flatter a
    model for correctly dismissing the overwhelming majority of empty cells.

``precision@k`` / ``recall@k``
    The question an operator actually asks: if I can inspect k street-hours,
    how many real crashes do I catch? This is the metric the map is for.

``lift@k``
    precision@k divided by the base rate -- how many times better than random.

Every metric is computed on the complete, unsampled test panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config


@dataclass
class ScoredPanel:
    """Accumulates scores over the test panel without holding every chunk."""

    y_true: list[np.ndarray] = field(default_factory=list)
    y_score: list[np.ndarray] = field(default_factory=list)

    def add(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        self.y_true.append(np.asarray(y_true, dtype=np.int8))
        self.y_score.append(np.asarray(y_score, dtype=np.float64))

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(self.y_true), np.concatenate(self.y_score)


def precision_recall_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k: int
) -> tuple[float, float]:
    """Precision and recall among the k highest-scoring cells."""
    k = min(k, y_score.size)
    if k == 0:
        return float("nan"), float("nan")
    top = np.argpartition(-y_score, k - 1)[:k]
    hits = int(y_true[top].sum())
    total_positive = int(y_true.sum())
    precision = hits / k
    recall = hits / total_positive if total_positive else float("nan")
    return precision, recall


def evaluate_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    name: str,
    ks: tuple[int, ...] = config.PRECISION_AT_K,
) -> dict[str, float | str]:
    """Full metric set for one scorer over the unsampled panel."""
    base_rate = float(y_true.mean())
    row: dict[str, float | str] = {
        "model": name,
        "n_cells": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "base_rate": base_rate,
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }
    for k in ks:
        precision, recall = precision_recall_at_k(y_true, y_score, k)
        row[f"precision@{k}"] = precision
        row[f"recall@{k}"] = recall
        row[f"lift@{k}"] = precision / base_rate if base_rate else float("nan")
    return row


def calibration_table(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10
) -> pd.DataFrame:
    """Predicted vs observed rate, by quantile bin.

    This is what earns a score the word "probability". A well-calibrated model
    has ``mean_predicted`` close to ``observed_rate`` in every row.
    """
    ranks = pd.Series(y_prob).rank(method="first")
    bins = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop")
    table = (
        pd.DataFrame({"bin": bins, "y": y_true, "p": y_prob})
        .groupby("bin", observed=True)
        .agg(
            n=("y", "size"),
            mean_predicted=("p", "mean"),
            observed_rate=("y", "mean"),
        )
        .reset_index(drop=True)
    )
    return table


# --- baselines -------------------------------------------------------------
# A model is only interesting if it beats the obvious alternatives. v1 compared
# against nothing.


def baseline_random(frame: pd.DataFrame, *, seed: int = config.SEED) -> np.ndarray:
    """Uniform noise -- the floor any model must clear."""
    rng = np.random.default_rng(seed)
    return rng.random(len(frame))


def baseline_segment_length(frame: pd.DataFrame) -> np.ndarray:
    """Longer segments contain more road, so they should see more crashes."""
    return np.nan_to_num(frame["segmentlength"].to_numpy(), nan=0.0)


def baseline_historical(frame: pd.DataFrame) -> np.ndarray:
    """Crashes per segment in the training window.

    The strong baseline, and the honest one: if a model cannot beat "this
    street had crashes before", it has learned nothing about time.
    """
    return np.nan_to_num(frame["hist_crash_count"].to_numpy(), nan=0.0)


BASELINES = {
    "baseline_random": baseline_random,
    "baseline_segment_length": baseline_segment_length,
    "baseline_historical": baseline_historical,
}
