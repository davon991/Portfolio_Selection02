from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def sigma_p(x: np.ndarray, V: np.ndarray) -> float:
    val = float(x @ V @ x)
    return float(np.sqrt(max(val, 0.0)))


def ctr_vector(x: np.ndarray, V: np.ndarray) -> np.ndarray:
    denom = float(x @ V @ x)
    if denom <= 0:
        raise ValueError('Portfolio variance must be positive.')
    return (x * (V @ x)) / denom


def ctb_vector(x: np.ndarray, V: np.ndarray) -> np.ndarray:
    sigmas = np.sqrt(np.maximum(np.diag(V), 1e-18))
    sp = max(sigma_p(x, V), 1e-18)
    return (V @ x) / (sigmas * sp)


def D_R(x: np.ndarray, V: np.ndarray, b: np.ndarray) -> float:
    ctr = ctr_vector(x, V)
    return float(np.sum((ctr - b) ** 2))


def D_B(x: np.ndarray, V: np.ndarray) -> float:
    ctb = ctb_vector(x, V)
    ctb_mean = float(ctb.mean())
    return float(np.sum((ctb - ctb_mean) ** 2))


def objective_terms_rb_ctb_band(
    x: np.ndarray,
    V: np.ndarray,
    b: np.ndarray,
    x_prev: np.ndarray,
    delta: float,
    eta: float,
    gamma: float,
    rho: float,
) -> dict[str, float]:
    dr_term = D_R(x, V, b)
    smooth_term = float(eta * np.sum((x - x_prev) ** 2))
    l2_term = float(gamma * np.sum(x ** 2))
    db = D_B(x, V)
    band_penalty = float(0.5 * rho * max(db - delta, 0.0) ** 2)
    return {
        'obj_total': dr_term + smooth_term + l2_term + band_penalty,
        'dr_term': dr_term,
        'smooth_term': smooth_term,
        'l2_term': l2_term,
        'band_penalty': band_penalty,
        'db': db,
    }


def rebalance_turnover(x_new: np.ndarray, x_prev: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(x_new - x_prev)))


def drift_weights(x_target: np.ndarray, future_returns: pd.DataFrame) -> np.ndarray:
    if future_returns.empty:
        return x_target.copy()
    gross = (1.0 + future_returns).prod(axis=0).to_numpy(dtype=float)
    w = x_target * gross
    s = float(w.sum())
    if s <= 0:
        return x_target.copy()
    return w / s


def annual_metrics(ret: pd.Series, ann_factor: int = 252) -> dict[str, float]:
    s = pd.Series(ret).dropna().astype(float)
    if s.empty:
        return {'ann_return': np.nan, 'ann_vol': np.nan, 'sharpe': np.nan, 'max_drawdown': np.nan}
    nav = (1.0 + s).cumprod()
    total = float(nav.iloc[-1])
    n = len(s)
    ann_return = total ** (ann_factor / max(n, 1)) - 1.0
    ann_vol = float(s.std(ddof=1)) * np.sqrt(ann_factor) if n > 1 else np.nan
    sharpe = ann_return / ann_vol if ann_vol and ann_vol > 0 else np.nan
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    max_dd = float(drawdown.min())
    return {
        'ann_return': float(ann_return),
        'ann_vol': float(ann_vol),
        'sharpe': float(sharpe),
        'max_drawdown': max_dd,
    }


def panel_long_from_outputs(
    strategy_name: str,
    weights_df: pd.DataFrame,
    ctr_df: pd.DataFrame,
    ctb_df: pd.DataFrame,
    drdb_df: pd.DataFrame,
) -> pd.DataFrame:
    w_long = weights_df.melt(id_vars=['date'], var_name='asset', value_name='weight')
    ctr_long = ctr_df.melt(id_vars=['date'], var_name='asset', value_name='ctr')
    ctb_long = ctb_df.melt(id_vars=['date'], var_name='asset', value_name='ctb')

    out = w_long.merge(ctr_long, on=['date', 'asset'], how='left').merge(ctb_long, on=['date', 'asset'], how='left')
    out = out.merge(drdb_df[['date', 'dr', 'db', 'band_active', 'turnover', 'converged', 'solver_status', 'nit']], on='date', how='left')
    out['strategy'] = strategy_name
    cols = ['date', 'strategy', 'asset', 'weight', 'ctr', 'ctb', 'dr', 'db', 'band_active', 'turnover', 'converged', 'solver_status', 'nit']
    return out[cols]