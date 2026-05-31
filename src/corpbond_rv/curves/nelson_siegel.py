from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NelsonSiegelFit:
    beta0: float
    beta1: float
    beta2: float
    tau: float
    rmse: float
    n_obs: int

    def predict(self, maturity_years: np.ndarray) -> np.ndarray:
        return nelson_siegel(maturity_years, self.beta0, self.beta1, self.beta2, self.tau)


def _as_positive_array(x: np.ndarray | list[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(arr <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return arr


def factor_matrix(maturity_years: np.ndarray | list[float], tau: float) -> np.ndarray:
    """Nelson-Siegel loading matrix [level, slope, curvature]."""
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("tau must be positive and finite")
    t = _as_positive_array(maturity_years, "maturity_years")
    x = t / tau
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (1.0 - np.exp(-x)) / x
    slope = np.where(np.isclose(x, 0.0), 1.0, slope)
    curvature = slope - np.exp(-x)
    return np.column_stack([np.ones_like(t), slope, curvature])


def nelson_siegel(
    maturity_years: np.ndarray | list[float],
    beta0: float,
    beta1: float,
    beta2: float,
    tau: float,
) -> np.ndarray:
    """Evaluate the Nelson-Siegel curve."""
    beta = np.array([beta0, beta1, beta2], dtype=float)
    return factor_matrix(maturity_years, tau) @ beta


def fit_nelson_siegel(
    maturity_years: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    weights: np.ndarray | list[float] | None = None,
    tau_grid: np.ndarray | list[float] | None = None,
) -> NelsonSiegelFit:
    """Fit Nelson-Siegel by grid-searching tau and solving weighted least squares."""
    t = _as_positive_array(maturity_years, "maturity_years")
    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or values.shape[0] != t.shape[0]:
        raise ValueError("y must be one-dimensional and match maturity_years")
    if np.any(~np.isfinite(values)):
        raise ValueError("y contains non-finite values")
    if t.shape[0] < 3:
        raise ValueError("Need at least 3 observations to fit Nelson-Siegel")

    if tau_grid is None:
        tau_grid = np.geomspace(0.25, 20.0, 80)
    tau_values = _as_positive_array(tau_grid, "tau_grid")

    if weights is None:
        w = np.ones_like(values)
    else:
        w = _as_positive_array(weights, "weights")
        if w.shape[0] != values.shape[0]:
            raise ValueError("weights must match y")

    sqrt_w = np.sqrt(w)
    best: NelsonSiegelFit | None = None

    for tau in tau_values:
        x = factor_matrix(t, float(tau))
        xw = x * sqrt_w[:, None]
        yw = values * sqrt_w
        beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
        pred = x @ beta
        rmse = float(np.sqrt(np.average((values - pred) ** 2, weights=w)))
        fit = NelsonSiegelFit(
            beta0=float(beta[0]),
            beta1=float(beta[1]),
            beta2=float(beta[2]),
            tau=float(tau),
            rmse=rmse,
            n_obs=int(t.shape[0]),
        )
        if best is None or fit.rmse < best.rmse:
            best = fit

    assert best is not None
    return best
