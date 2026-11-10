"""The risk model: predict disengagement, and give calibrated probabilities.

Two models are provided:
- RiskModel: a gradient-boosted tree model (LightGBM). It is the main model.
- LogisticModel: a simple logistic-regression baseline, to compare against.

Both are calibrated. A risk score is only useful if it means what it says: of
the students given "70% risk", about 70% should really disengage. We fit the
model on one slice of the training data and an isotonic calibrator on another,
so the calibration is not learned on the same rows the model saw.

RiskModel can also return, for one student, how much each feature pushed their
risk up or down (LightGBM's SHAP values). That is what the plain-English
explanation is built from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _split(n, frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * frac)
    return idx[cut:], idx[:cut]   # fit_idx, calib_idx


@dataclass
class RiskModel:
    model: LGBMClassifier
    iso: IsotonicRegression
    names: list[str]
    name: str = "LightGBM (calibrated)"

    @classmethod
    def train(cls, X, y, names, seed=0):
        fit_idx, cal_idx = _split(len(X), 0.25, seed)
        model = LGBMClassifier(n_estimators=200, num_leaves=16, learning_rate=0.05,
                               min_child_samples=15, subsample=0.9, colsample_bytree=0.9,
                               random_state=seed, verbose=-1)
        model.fit(X[fit_idx], y[fit_idx])
        raw = model.predict_proba(X[cal_idx])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y[cal_idx])
        return cls(model, iso, list(names))

    def predict_proba(self, X):
        return self.iso.transform(self.model.predict_proba(X)[:, 1])

    def contributions(self, x_row):
        """SHAP contributions for one student: {feature: signed value}, plus base."""
        c = self.model.predict(x_row.reshape(1, -1), pred_contrib=True)[0]
        contrib = {n: float(c[i]) for i, n in enumerate(self.names)}
        return contrib, float(c[-1])


@dataclass
class LogisticModel:
    scaler: StandardScaler
    model: LogisticRegression
    iso: IsotonicRegression
    names: list[str]
    name: str = "logistic regression (calibrated)"

    @classmethod
    def train(cls, X, y, names, seed=0):
        fit_idx, cal_idx = _split(len(X), 0.25, seed)
        scaler = StandardScaler().fit(X[fit_idx])
        model = LogisticRegression(max_iter=1000).fit(scaler.transform(X[fit_idx]), y[fit_idx])
        raw = model.predict_proba(scaler.transform(X[cal_idx]))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y[cal_idx])
        return cls(scaler, model, iso, list(names))

    def predict_proba(self, X):
        return self.iso.transform(self.model.predict_proba(self.scaler.transform(X))[:, 1])

    def contributions(self, x_row):
        """Exact per-student contributions: weight x standardised value.

        A linear model makes this exact, not an approximation: each feature's
        contribution is its coefficient times how far the student sits from the
        average on that feature. Positive pushes risk up."""
        z = (x_row - self.scaler.mean_) / self.scaler.scale_
        contrib = {n: float(self.model.coef_[0][i] * z[i]) for i, n in enumerate(self.names)}
        return contrib, float(self.model.intercept_[0])
