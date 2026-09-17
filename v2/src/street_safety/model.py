"""Models, and the correction that makes their output mean something.

The headline output of this project is a **ranking**: which street-hours are
riskiest relative to each other. Ranking is invariant to any monotone transform
of the score, so it survives case-control sampling untouched -- which is why
v1's map was not completely worthless even though its numbers were.

Absolute probabilities do *not* survive. Fitting on a sample with a 9%
positive rate when the population rate is 0.01% inflates every predicted
probability by roughly three orders of magnitude. For logistic regression the
distortion is confined entirely to the intercept, and King & Zeng (2001) give
the closed-form correction implemented in ``prior_corrected_intercept``.

That is the whole v1/v2 difference in one function: v1 printed
"Accident probability: 0.899" from an uncorrected, non-probabilistic model.
Here, a score is only ever called a probability after the correction, and
``evaluate.calibration_table`` checks whether it earned the name.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .features import make_preprocessor


def prior_corrected_intercept(
    intercept: float, tau: float, ybar: float
) -> float:
    """Shift a logistic intercept from the sample prior back to the population.

    King & Zeng (2001), "Logistic Regression in Rare Events Data":

        beta0_corrected = beta0 - ln[ ((1 - tau) / tau) * (ybar / (1 - ybar)) ]

    where ``tau`` is the population positive rate and ``ybar`` the rate in the
    sample. Slopes are unaffected, so the ranking is unchanged -- only the
    absolute level moves.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    if not 0.0 < ybar < 1.0:
        raise ValueError(f"ybar must be in (0, 1), got {ybar}")
    return float(intercept - np.log(((1 - tau) / tau) * (ybar / (1 - ybar))))


@dataclass
class FittedModel:
    """A fitted pipeline plus the sampling facts needed to interpret it."""

    name: str
    pipeline: Pipeline
    tau: float
    sample_rate: float
    corrected_intercept: float | None = None

    def rank_scores(self, X) -> np.ndarray:
        """Scores for ranking. Monotone in risk; not probabilities."""
        return self.pipeline.predict_proba(X)[:, 1]

    def population_probabilities(self, X) -> np.ndarray | None:
        """Calibrated population-scale probabilities, where available.

        Returns ``None`` for models whose intercept cannot be corrected in
        closed form -- honest refusal rather than a plausible-looking number.
        """
        if self.corrected_intercept is None:
            return None
        clf = self.pipeline.named_steps["clf"]
        design = self.pipeline.named_steps["prep"].transform(X)
        logits = design @ clf.coef_.ravel() + self.corrected_intercept
        return 1.0 / (1.0 + np.exp(-logits))


def fit_logistic(X, y, *, tau: float, sample_rate: float) -> FittedModel:
    """Logistic regression, with the intercept corrected back to scale."""
    pipe = Pipeline(
        [
            ("prep", make_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=config.SEED,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    ybar = float(np.mean(y))
    corrected = prior_corrected_intercept(
        float(pipe.named_steps["clf"].intercept_[0]), tau, ybar
    )
    return FittedModel(
        name="logistic_regression",
        pipeline=pipe,
        tau=tau,
        sample_rate=sample_rate,
        corrected_intercept=corrected,
    )


def fit_gradient_boosting(X, y, *, tau: float, sample_rate: float) -> FittedModel:
    """Gradient boosting, for ranking only.

    No closed-form prior correction exists for this, so
    ``population_probabilities`` returns ``None`` rather than a number that
    would look calibrated without being so.
    """
    pipe = Pipeline(
        [
            ("prep", make_preprocessor()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.1,
                    max_leaf_nodes=31,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=config.SEED,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return FittedModel(
        name="gradient_boosting",
        pipeline=pipe,
        tau=tau,
        sample_rate=sample_rate,
        corrected_intercept=None,
    )
