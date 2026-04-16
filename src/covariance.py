from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def estimate_covariance(returns_window: np.ndarray, method: str) -> np.ndarray:
    if returns_window.ndim != 2:
        raise ValueError('returns_window must be 2D (L, n)')
    if method == 'sample':
        return np.cov(returns_window, rowvar=False, ddof=1)
    if method == 'ledoit_wolf_shrinkage':
        return LedoitWolf().fit(returns_window).covariance_
    raise ValueError(f'Unknown covariance_method: {method}')


def covariance_to_correlation(V: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.maximum(np.diag(V), 0.0))
    denom = np.outer(sigma, sigma)
    C = np.divide(V, denom, out=np.zeros_like(V), where=denom > 0)
    np.fill_diagonal(C, 1.0)
    return C


def covariance_to_vol(V: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.diag(V), 0.0))