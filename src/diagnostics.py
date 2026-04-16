from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import longest_false_streak


def build_diagnostics(cfg: dict[str, Any], strategy_outputs: dict[str, dict[str, Any]], delta_star: float | None, eps_db: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        'run_id': None,
        'strategy_health': {},
        'solve_success_rate': {},
        'fallback_rate': {},
        'active_rate': {},
        'kkt_residual_stats': {},
        'constraint_violation_stats': {},
        'band_violation_stats': {},
        'step_rejection_stats': {},
        'warning_dates': {},
        'failed_dates': {},
    }
    for name, payload in strategy_outputs.items():
        drdb: pd.DataFrame = payload.drdb if hasattr(payload, 'drdb') else payload['drdb']
        if drdb.empty:
            out['strategy_health'][name] = {'empty': True}
            continue
        converged = drdb['converged'].astype(bool)
        success_rate = float(converged.mean())
        active_rate = float(drdb['band_active'].mean()) if 'band_active' in drdb.columns else 0.0
        max_fail_streak = int(longest_false_streak(converged.tolist()))
        failed_dates = drdb.loc[~converged, 'date'].astype(str).tolist()
        out['strategy_health'][name] = {
            'n_rebalances': int(len(drdb)),
            'success_rate': success_rate,
            'active_rate': active_rate,
            'max_fail_streak': max_fail_streak,
            'turnover_mean': float(drdb['turnover'].mean()),
            'turnover_p95': float(drdb['turnover'].quantile(0.95)),
        }
        out['solve_success_rate'][name] = success_rate
        out['active_rate'][name] = active_rate
        out['failed_dates'][name] = failed_dates
        out['warning_dates'][name] = failed_dates
        if delta_star is not None and 'db' in drdb.columns:
            vio = np.maximum(drdb['db'].to_numpy(dtype=float) - delta_star, 0.0)
            out['band_violation_stats'][name] = {
                'mean': float(np.mean(vio)),
                'max': float(np.max(vio)),
            }
        else:
            out['band_violation_stats'][name] = {'mean': 0.0, 'max': 0.0}
        out['constraint_violation_stats'][name] = {'budget_abs_max': float(drdb.get('budget_error', pd.Series([0.0])).abs().max())}
        out['kkt_residual_stats'][name] = {'mean': float(drdb.get('kkt_residual', pd.Series([np.nan])).mean())}
        out['step_rejection_stats'][name] = {'not_available_in_stage_9A': True}
    return out