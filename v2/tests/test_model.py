"""Tests for the prior correction -- the mathematical heart of v2.

The central claim of this rebuild is that case-control sampling is fine *if you
correct for it*. These tests demonstrate that on synthetic data where the true
population rate is known by construction, so the claim is checked rather than
asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from street_safety.model import prior_corrected_intercept


def test_no_correction_when_sample_matches_population():
    """If the sample rate already equals the population rate, nothing moves."""
    assert prior_corrected_intercept(0.7, tau=0.25, ybar=0.25) == pytest.approx(0.7)


def test_correction_lowers_intercept_when_positives_oversampled():
    """Oversampling positives inflates predictions, so the shift is negative."""
    corrected = prior_corrected_intercept(0.0, tau=0.001, ybar=0.5)
    assert corrected < 0.0


def test_correction_magnitude_matches_king_and_zeng():
    tau, ybar, beta0 = 0.0001, 0.1, -1.3
    expected = beta0 - np.log(((1 - tau) / tau) * (ybar / (1 - ybar)))
    assert prior_corrected_intercept(beta0, tau, ybar) == pytest.approx(expected)


@pytest.mark.parametrize(
    "tau,ybar", [(0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0), (-0.1, 0.5)]
)
def test_correction_rejects_degenerate_rates(tau, ybar):
    with pytest.raises(ValueError):
        prior_corrected_intercept(0.0, tau, ybar)


def test_correction_recovers_population_rate_on_synthetic_data():
    """The end-to-end claim.

    Build a population with a known rare-event rate, case-control sample it,
    fit on the sample, correct the intercept, and check that predictions made
    on the *population* average back to the true rate.

    Without the correction the predictions are off by orders of magnitude --
    which is precisely v1's failure.
    """
    rng = np.random.default_rng(0)
    n = 400_000
    x = rng.normal(size=n)
    # Intercept chosen to make positives genuinely rare.
    true_logit = -9.0 + 1.5 * x
    p = 1.0 / (1.0 + np.exp(-true_logit))
    y = rng.binomial(1, p)

    tau = y.mean()
    assert 0.0001 < tau < 0.01, f"fixture should be rare, got {tau}"

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = rng.choice(np.flatnonzero(y == 0), size=pos_idx.size * 10, replace=False)
    sample_idx = np.concatenate([pos_idx, neg_idx])

    X_s = x[sample_idx].reshape(-1, 1)
    y_s = y[sample_idx]
    ybar = y_s.mean()

    clf = LogisticRegression(max_iter=1000).fit(X_s, y_s)

    uncorrected_mean = clf.predict_proba(x.reshape(-1, 1))[:, 1].mean()
    corrected_b0 = prior_corrected_intercept(float(clf.intercept_[0]), tau, ybar)
    corrected_mean = float(
        np.mean(1.0 / (1.0 + np.exp(-(x * clf.coef_.ravel()[0] + corrected_b0))))
    )

    # Uncorrected output is wildly inflated ...
    assert uncorrected_mean > tau * 20
    # ... corrected output lands close to the truth.
    assert corrected_mean == pytest.approx(tau, rel=0.25)


def test_correction_preserves_ranking():
    """Ranking is invariant to the correction: only the level changes.

    This is why v1's *map* retained some value even though its *numbers* did
    not -- and why v2 reports ranking metrics as the headline.
    """
    rng = np.random.default_rng(1)
    logits = rng.normal(size=500)
    corrected = logits + prior_corrected_intercept(0.0, tau=0.0005, ybar=0.2)

    original_order = pd.Series(logits).rank(method="first").to_numpy()
    corrected_order = pd.Series(corrected).rank(method="first").to_numpy()
    np.testing.assert_array_equal(original_order, corrected_order)
