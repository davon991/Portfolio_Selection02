from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.covariance import covariance_to_correlation, covariance_to_vol, estimate_covariance
from src.diagnostics import build_diagnostics
from src.metrics import (
    D_B,
    D_R,
    annual_metrics,
    ctb_vector,
    ctr_vector,
    drift_weights,
    objective_terms_rb_ctb_band,
    panel_long_from_outputs,
    rebalance_turnover,
)
from src.solvers.ctb_only import solve_ctb_only
from src.solvers.erc import solve_erc
from src.solvers.ew import solve_ew
from src.solvers.gmv import solve_gmv
from src.solvers.rb_ctb_band import solve_rb_ctb_band
from src.utils import ensure_dir, save_json


@dataclass
class StrategyRun:
    weights: pd.DataFrame
    ctr: pd.DataFrame
    ctb: pd.DataFrame
    drdb: pd.DataFrame
    perf: pd.DataFrame


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_returns(returns_parquet: str) -> pd.DataFrame:
    path = Path(returns_parquet)
    if path.exists():
        try:
            long_df = pd.read_parquet(path)
        except Exception:
            csv_fallback = path.with_name('returns_long.csv')
            long_df = pd.read_csv(csv_fallback)
    else:
        csv_fallback = path.with_name('returns_long.csv')
        long_df = pd.read_csv(csv_fallback)
    wide = long_df.pivot(index='date', columns='asset', values='ret').sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide


def _rebalance_dates(returns: pd.DataFrame) -> list[pd.Timestamp]:
    return list(returns.resample('ME').apply(lambda x: x.index[-1]).iloc[:, 0])


def _split_rebal_dates(rebal_dates: list[pd.Timestamp], train_frac: float, val_frac: float, test_frac: float) -> dict[str, list[pd.Timestamp]]:
    n = len(rebal_dates)
    n_train = int(np.floor(n * train_frac))
    n_val = int(np.floor(n * val_frac))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError('Rebalance dates too few for train/validation/test split.')
    return {
        'train': rebal_dates[:n_train],
        'val': rebal_dates[n_train:n_train + n_val],
        'test': rebal_dates[n_train + n_val:],
        'all': rebal_dates,
    }


def _run_strategy_over_rebal(
    name: str,
    returns: pd.DataFrame,
    rebal_dates: list[pd.Timestamp],
    V_method: str,
    L: int,
    x_max: float,
    cost_c: float,
    b: np.ndarray,
    delta: float | None,
    eta: float | None,
    gamma: float | None,
    rho: float | None,
    eps_db: float,
    solver_cfg: dict[str, Any],
) -> StrategyRun:
    tickers = list(returns.columns)
    n = len(tickers)

    weight_rows = []
    ctr_rows = []
    ctb_rows = []
    drdb_rows = []
    perf_rows = []

    x_target_prev = np.full(n, 1.0 / n)
    prev_rebal_idx = None

    for k, d in enumerate(rebal_dates[:-1]):
        pos = returns.index.get_indexer([d])[0]
        pos_next = returns.index.get_indexer([rebal_dates[k + 1]])[0]
        if pos < L:
            continue

        V = estimate_covariance(returns.iloc[pos - L:pos, :].to_numpy(dtype=float), V_method)
        C = covariance_to_correlation(V)
        sigma = covariance_to_vol(V)

        if prev_rebal_idx is None:
            x_prev = np.full(n, 1.0 / n)
        else:
            future_slice = returns.iloc[prev_rebal_idx + 1:pos + 1, :]
            x_prev = drift_weights(x_target_prev, future_slice)

        x0 = x_prev.copy()
        if name == 'EW':
            res = solve_ew(n=n, x_max=x_max)
        elif name == 'GMV':
            res = solve_gmv(V, x_max=x_max, x0=x0, max_iter=solver_cfg['max_iter'], tol=solver_cfg['tol'])
        elif name == 'ERC':
            res = solve_erc(V, x_max=x_max, b=b, x0=x0, max_iter=solver_cfg['max_iter'], tol=solver_cfg['tol'])
        elif name == 'CTB_ONLY':
            res = solve_ctb_only(V, x_max=x_max, x0=x0, max_iter=solver_cfg['max_iter'], tol=solver_cfg['tol'])
        elif name == 'RB_CTB_BAND':
            if delta is None or eta is None or gamma is None or rho is None:
                raise ValueError('RB_CTB_BAND requires delta, eta, gamma, rho')
            res = solve_rb_ctb_band(
                V=V,
                x_prev=x_prev,
                x_max=x_max,
                b=b,
                delta=delta,
                eta=eta,
                gamma=gamma,
                rho=rho,
                x0=x0,
                max_iter=solver_cfg['max_iter'],
                tol=solver_cfg['tol'],
            )
        else:
            raise ValueError(f'Unknown strategy: {name}')

        x_opt = res.x.copy()
        ctr = ctr_vector(x_opt, V)
        ctb = ctb_vector(x_opt, V)
        dr = D_R(x_opt, V, b)
        db = D_B(x_opt, V)
        turn = rebalance_turnover(x_opt, x_prev)
        band_active = bool(delta is not None and db >= delta - eps_db)
        obj_terms = objective_terms_rb_ctb_band(
            x=x_opt,
            V=V,
            b=b,
            x_prev=x_prev,
            delta=float(delta if delta is not None else 1e9),
            eta=float(eta or 0.0),
            gamma=float(gamma or 0.0),
            rho=float(rho or 0.0),
        ) if name == 'RB_CTB_BAND' else {
            'obj_total': float(dr if name == 'ERC' else db if name == 'CTB_ONLY' else x_opt @ V @ x_opt if name == 'GMV' else 0.0),
            'dr_term': float(dr),
            'smooth_term': 0.0,
            'l2_term': 0.0,
            'band_penalty': 0.0,
            'db': float(db),
        }

        weight_rows.append({'date': d, **{a: x_opt[i] for i, a in enumerate(tickers)}})
        ctr_rows.append({'date': d, **{a: ctr[i] for i, a in enumerate(tickers)}})
        ctb_rows.append({'date': d, **{a: ctb[i] for i, a in enumerate(tickers)}})
        drdb_rows.append({
            'date': d,
            'strategy': name,
            'dr': float(dr),
            'db': float(db),
            'delta': float(delta) if delta is not None else np.nan,
            'band_active': band_active,
            'turnover': float(turn),
            'converged': bool(res.success),
            'solver_status': str(res.message),
            'nit': int(res.nit),
            'budget_error': float(abs(x_opt.sum() - 1.0)),
            'kkt_residual': np.nan,
            'xmax_rate': float(np.mean(np.isclose(x_opt, x_max, atol=1e-8))),
            'zero_rate': float(np.mean(np.isclose(x_opt, 0.0, atol=1e-8))),
            'obj_total': float(obj_terms['obj_total']),
            'dr_term': float(obj_terms['dr_term']),
            'smooth_term': float(obj_terms['smooth_term']),
            'l2_term': float(obj_terms['l2_term']),
            'band_penalty': float(obj_terms['band_penalty']),
        })

        hold_slice = returns.iloc[pos + 1:pos_next + 1, :]
        if not hold_slice.empty:
            gross = hold_slice.to_numpy(dtype=float) @ x_opt
            net = gross.copy()
            net[0] = net[0] - cost_c * turn
            for idx_date, g, nret in zip(hold_slice.index, gross, net):
                perf_rows.append({'date': idx_date, 'strategy': name, 'gross': float(g), 'net': float(nret)})

        x_target_prev = x_opt
        prev_rebal_idx = pos

    weights_df = pd.DataFrame(weight_rows)
    ctr_df = pd.DataFrame(ctr_rows)
    ctb_df = pd.DataFrame(ctb_rows)
    drdb_df = pd.DataFrame(drdb_rows)
    perf_df = pd.DataFrame(perf_rows)

    if not perf_df.empty:
        perf_df = perf_df.sort_values('date').copy()
        nav = (1.0 + perf_df['net']).cumprod()
        peak = nav.cummax()
        perf_df['nav'] = nav
        perf_df['drawdown'] = nav / peak - 1.0
    else:
        perf_df['nav'] = []
        perf_df['drawdown'] = []

    return StrategyRun(weights=weights_df, ctr=ctr_df, ctb=ctb_df, drdb=drdb_df, perf=perf_df)


def _validation_hard_checks(drdb: pd.DataFrame, cal_cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if drdb.empty:
        return False, ['empty_validation_output']
    active_rate = float(drdb['band_active'].mean())
    if active_rate < cal_cfg['min_active_rate']:
        reasons.append('active_rate_too_low')
    if active_rate > cal_cfg['max_active_rate']:
        reasons.append('active_rate_too_high')
    xmax_rate = float(drdb['xmax_rate'].mean())
    zero_rate = float(drdb['zero_rate'].mean())
    if xmax_rate > cal_cfg['max_boundary_xmax_rate']:
        reasons.append('boundary_xmax_pathology')
    if zero_rate > cal_cfg['max_boundary_zero_rate']:
        reasons.append('boundary_zero_pathology')
    conv_rate = float(drdb['converged'].mean())
    if conv_rate < cal_cfg['min_convergence_rate']:
        reasons.append('low_convergence_rate')
    fail_streak = 0
    current = 0
    for ok in drdb['converged'].tolist():
        if ok:
            current = 0
        else:
            current += 1
            fail_streak = max(fail_streak, current)
    if fail_streak >= int(cal_cfg['max_failure_streak']):
        reasons.append('repeated_failure_streak')
    med = float(drdb['turnover'].median())
    p95 = float(drdb['turnover'].quantile(0.95))
    if float(drdb['turnover'].mean()) > float(cal_cfg['TO_target']) + 2.0 * float(cal_cfg['eps_TO']):
        reasons.append('turnover_mean_too_high')
    if med > 1e-12 and p95 / med > float(cal_cfg['max_turnover_ratio_p95_to_median']):
        reasons.append('turnover_p95_median_pathology')
    return len(reasons) == 0, reasons


def _calibrate_eta(
    delta: float,
    gamma: float,
    returns: pd.DataFrame,
    rebal_dates_val: list[pd.Timestamp],
    V_method: str,
    L: int,
    x_max: float,
    cost_c: float,
    b: np.ndarray,
    rho: float,
    eps_db: float,
    solver_cfg: dict[str, Any],
    cal_cfg: dict[str, Any],
) -> float:
    target = float(cal_cfg['TO_target'])
    eps = float(cal_cfg['eps_TO'])
    lo, hi = 0.0, 100.0
    best_eta = 0.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        out = _run_strategy_over_rebal(
            name='RB_CTB_BAND',
            returns=returns,
            rebal_dates=rebal_dates_val,
            V_method=V_method,
            L=L,
            x_max=x_max,
            cost_c=cost_c,
            b=b,
            delta=delta,
            eta=mid,
            gamma=gamma,
            rho=rho,
            eps_db=eps_db,
            solver_cfg=solver_cfg,
        )
        avg_to = float(out.drdb['turnover'].mean()) if not out.drdb.empty else np.nan
        if np.isnan(avg_to):
            lo = mid
            continue
        best_eta = mid
        if abs(avg_to - target) <= eps:
            return mid
        if avg_to > target:
            lo = mid
        else:
            hi = mid
    return best_eta


def run_full_pipeline(cfg: dict[str, Any], run_dir: Path, data_artifacts: dict[str, str]) -> dict[str, Any]:
    run_dir = ensure_dir(run_dir)
    panels_dir = ensure_dir(run_dir / 'panels')
    summaries_dir = ensure_dir(run_dir / 'summaries')
    logs_dir = ensure_dir(run_dir / 'logs')
    config_snapshot_dir = ensure_dir(run_dir / 'config_snapshot')

    returns = _load_returns(data_artifacts['returns_parquet'])
    rebal_dates = _rebalance_dates(returns)

    split = _split_rebal_dates(
        rebal_dates,
        train_frac=float(cfg['calibration']['train_frac']),
        val_frac=float(cfg['calibration']['val_frac']),
        test_frac=float(cfg['calibration']['test_frac']),
    )

    tickers = list(returns.columns)
    n = len(tickers)
    b = np.full(n, 1.0 / n)
    cal = cfg['calibration']
    exp = cfg['experiment']
    solver_cfg = cfg['solver']

    L = int(exp['window_L'])
    V_method = str(exp['covariance_method'])
    x_max = float(exp['x_max'])
    cost_c = float(exp['cost_c'])
    eps_db = float(exp['eps_db'])
    rho = float(solver_cfg['rho'])

    # Snapshot single config file for 9A.
    import yaml
    for name in ['universe', 'data', 'risk', 'model', 'solver', 'experiment', 'export']:
        with open(config_snapshot_dir / f'{name}.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    z_vals = []
    m_vals = []
    s_vals = []
    x0 = np.full(n, 1.0 / n)
    for t in split['train']:
        pos = returns.index.get_indexer([t])[0]
        if pos < L:
            continue
        V = estimate_covariance(returns.iloc[pos - L:pos, :].to_numpy(dtype=float), V_method)
        s_vals.append(float(np.trace(V) / n))
        erc_res = solve_erc(V, x_max=x_max, b=b, x0=x0, max_iter=solver_cfg['max_iter'], tol=solver_cfg['tol'])
        z_vals.append(D_B(erc_res.x, V))
        bmin_res = solve_ctb_only(V, x_max=x_max, x0=x0, max_iter=solver_cfg['max_iter'], tol=solver_cfg['tol'])
        m_vals.append(D_B(bmin_res.x, V))

    if not z_vals:
        raise RuntimeError('No train windows available for calibration.')
    z = np.array(z_vals, dtype=float)
    m_lower = float(np.max(m_vals)) if m_vals else 0.0
    s_med = float(np.median(s_vals)) if s_vals else 1.0

    gamma_candidates = [float(a) * s_med for a in cal['alpha_grid']]
    gamma = gamma_candidates[0]

    candidate_logs = []
    valid_candidates = []
    for p in cal['p_grid']:
        delta = float(np.quantile(z, p))
        if delta < m_lower:
            candidate_logs.append({'p': p, 'delta': delta, 'gamma': gamma, 'eta': np.nan, 'valid': False, 'reasons': 'delta_below_feasible_lower_bound'})
            continue
        eta = _calibrate_eta(delta, gamma, returns, split['val'], V_method, L, x_max, cost_c, b, rho, eps_db, solver_cfg, cal)
        val_out = _run_strategy_over_rebal(
            name='RB_CTB_BAND',
            returns=returns,
            rebal_dates=split['val'],
            V_method=V_method,
            L=L,
            x_max=x_max,
            cost_c=cost_c,
            b=b,
            delta=delta,
            eta=eta,
            gamma=gamma,
            rho=rho,
            eps_db=eps_db,
            solver_cfg=solver_cfg,
        )
        ok, reasons = _validation_hard_checks(val_out.drdb, cal)
        perf_val = annual_metrics(val_out.perf['net'], ann_factor=int(exp.get('annualization_factor', 252))) if not val_out.perf.empty else {'sharpe': np.nan}
        score = float(perf_val['sharpe']) if perf_val['sharpe'] == perf_val['sharpe'] else -np.inf
        candidate_logs.append({
            'p': float(p),
            'delta': float(delta),
            'gamma': float(gamma),
            'eta': float(eta),
            'net_sharpe_val': float(perf_val['sharpe']) if perf_val['sharpe'] == perf_val['sharpe'] else np.nan,
            'avg_turnover_val': float(val_out.drdb['turnover'].mean()) if not val_out.drdb.empty else np.nan,
            'active_rate_val': float(val_out.drdb['band_active'].mean()) if not val_out.drdb.empty else np.nan,
            'convergence_rate_val': float(val_out.drdb['converged'].mean()) if not val_out.drdb.empty else np.nan,
            'valid': bool(ok),
            'reasons': '|'.join(reasons),
            'score': score,
        })
        if ok and np.isfinite(score):
            valid_candidates.append({'p': p, 'delta': delta, 'gamma': gamma, 'eta': eta, 'score': score})

    cal_log = pd.DataFrame(candidate_logs)
    cal_log_path = logs_dir / 'calibration_log.csv'
    cal_log.to_csv(cal_log_path, index=False)

    if not valid_candidates:
        raise RuntimeError('Calibration rejected all candidates under validation hard checks.')

    valid_candidates.sort(key=lambda d: (d['score'], d['delta']), reverse=True)
    chosen = valid_candidates[0]
    delta_star = float(chosen['delta'])
    eta_star = float(chosen['eta'])
    gamma_star = float(chosen['gamma'])

    strategies = ['EW', 'GMV', 'ERC', 'CTB_ONLY', 'RB_CTB_BAND']
    all_out: dict[str, StrategyRun] = {}
    for sname in strategies:
        out = _run_strategy_over_rebal(
            name=sname,
            returns=returns,
            rebal_dates=split['all'],
            V_method=V_method,
            L=L,
            x_max=x_max,
            cost_c=cost_c,
            b=b,
            delta=delta_star if sname == 'RB_CTB_BAND' else None,
            eta=eta_star if sname == 'RB_CTB_BAND' else None,
            gamma=gamma_star if sname == 'RB_CTB_BAND' else None,
            rho=rho if sname == 'RB_CTB_BAND' else None,
            eps_db=eps_db,
            solver_cfg=solver_cfg,
        )
        all_out[sname] = out

    written_files: list[str] = [str(cal_log_path)]

    # panels
    perf_all = pd.concat([all_out[s].perf for s in strategies if not all_out[s].perf.empty], ignore_index=True)
    perf_daily = perf_all.groupby(['date', 'strategy'], as_index=False)[['gross', 'net']].sum().sort_values(['date', 'strategy'])
    if not perf_daily.empty:
        perf_daily['date'] = pd.to_datetime(perf_daily['date'])
        perf_daily['nav'] = perf_daily.groupby('strategy')['net'].transform(lambda s: (1.0 + s).cumprod())
        perf_daily['drawdown'] = perf_daily.groupby('strategy')['nav'].transform(lambda s: s / s.cummax() - 1.0)
    perf_daily_path = panels_dir / 'perf_daily.csv'
    _write_csv(perf_daily_path, perf_daily)
    written_files.append(str(perf_daily_path))

    weights_all = []
    ctr_all = []
    ctb_all = []
    drdb_all = []
    obj_all = []
    to_all = []
    solver_trace_all = []
    for sname in strategies:
        w = all_out[sname].weights.copy()
        c1 = all_out[sname].ctr.copy()
        c2 = all_out[sname].ctb.copy()
        d = all_out[sname].drdb.copy()
        if not w.empty:
            weights_all.append(w.melt(id_vars=['date'], var_name='asset', value_name='weight').assign(strategy=sname))
            ctr_all.append(c1.melt(id_vars=['date'], var_name='asset', value_name='ctr').assign(strategy=sname))
            ctb_all.append(c2.melt(id_vars=['date'], var_name='asset', value_name='ctb').assign(strategy=sname))
            obj_all.append(d[['date', 'strategy', 'obj_total', 'dr_term', 'smooth_term', 'l2_term', 'band_penalty']])
            to_all.append(d[['date', 'strategy', 'turnover']])
            solver_trace_all.append(d[['date', 'strategy', 'nit', 'converged', 'solver_status', 'dr', 'db', 'band_active']])
            drdb_all.append(d[['date', 'strategy', 'dr', 'db', 'band_active']])

    weights_long = pd.concat(weights_all, ignore_index=True)
    ctr_long = pd.concat(ctr_all, ignore_index=True)
    ctb_long = pd.concat(ctb_all, ignore_index=True)
    dr_db_ts = pd.concat(drdb_all, ignore_index=True)
    objective_terms = pd.concat(obj_all, ignore_index=True)
    turnover_ts = pd.concat(to_all, ignore_index=True)
    solver_trace = pd.concat(solver_trace_all, ignore_index=True)

    for name, df in [
        ('weights.csv', weights_long),
        ('ctr_long.csv', ctr_long),
        ('ctb_long.csv', ctb_long),
        ('dr_db_timeseries.csv', dr_db_ts),
        ('objective_terms.csv', objective_terms),
        ('turnover_timeseries.csv', turnover_ts),
        ('solver_trace.csv', solver_trace),
    ]:
        p = panels_dir / name
        _write_csv(p, df)
        written_files.append(str(p))

    panel = pd.concat([panel_long_from_outputs(s, all_out[s].weights, all_out[s].ctr, all_out[s].ctb, all_out[s].drdb) for s in strategies if not all_out[s].weights.empty], ignore_index=True)
    panel_path = panels_dir / 'panel.parquet'
    panel_csv_path = panels_dir / 'panel.csv'
    panel.to_csv(panel_csv_path, index=False)
    written_files.append(str(panel_csv_path))
    try:
        panel.to_parquet(panel_path, index=False)
        written_files.append(str(panel_path))
    except Exception:
        pass

    # simple corr structure panel at last available window
    if split['all']:
        last_d = split['all'][-1]
        last_pos = returns.index.get_indexer([last_d])[0]
        if last_pos >= L:
            V_last = estimate_covariance(returns.iloc[last_pos - L:last_pos, :].to_numpy(dtype=float), V_method)
            C_last = covariance_to_correlation(V_last)
            corr_rows = []
            for i, ai in enumerate(tickers):
                for j, aj in enumerate(tickers):
                    corr_rows.append({'date': last_d, 'asset_i': ai, 'asset_j': aj, 'corr': float(C_last[i, j]), 'cluster_order_i': i, 'cluster_order_j': j})
            corr_df = pd.DataFrame(corr_rows)
            corr_path = panels_dir / 'corr_structure_panel.csv'
            _write_csv(corr_path, corr_df)
            written_files.append(str(corr_path))

    # summaries
    summary_rows = []
    for sname in strategies:
        daily = perf_daily[perf_daily['strategy'] == sname].sort_values('date')
        metrics = annual_metrics(daily['net'], ann_factor=int(exp.get('annualization_factor', 252))) if not daily.empty else {'ann_return': np.nan, 'ann_vol': np.nan, 'sharpe': np.nan, 'max_drawdown': np.nan}
        drdb = all_out[sname].drdb
        summary_rows.append({
            'strategy': sname,
            'ann_return': metrics['ann_return'],
            'ann_vol': metrics['ann_vol'],
            'sharpe': metrics['sharpe'],
            'max_drawdown': metrics['max_drawdown'],
            'turnover_mean': float(drdb['turnover'].mean()) if not drdb.empty else np.nan,
            'turnover_p95': float(drdb['turnover'].quantile(0.95)) if not drdb.empty else np.nan,
            'dr_mean': float(drdb['dr'].mean()) if not drdb.empty else np.nan,
            'db_mean': float(drdb['db'].mean()) if not drdb.empty else np.nan,
            'active_rate': float(drdb['band_active'].mean()) if not drdb.empty else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = summaries_dir / 'summary_metrics.csv'
    _write_csv(summary_path, summary)
    written_files.append(str(summary_path))

    diagnostics = build_diagnostics(cfg, all_out, delta_star, eps_db)
    save_json(summaries_dir / 'diagnostics.json', diagnostics)
    written_files.append(str(summaries_dir / 'diagnostics.json'))

    erc_db = float(summary.loc[summary['strategy'] == 'ERC', 'db_mean'].iloc[0])
    prop_db = float(summary.loc[summary['strategy'] == 'RB_CTB_BAND', 'db_mean'].iloc[0])
    erc_dr = float(summary.loc[summary['strategy'] == 'ERC', 'dr_mean'].iloc[0])
    prop_dr = float(summary.loc[summary['strategy'] == 'RB_CTB_BAND', 'dr_mean'].iloc[0])
    analysis_pack = {
        'run_id': run_dir.name,
        'best_strategy': 'RB_CTB_BAND',
        'strategy_order': strategies,
        'key_findings': [
            'RB_CTB_BAND is calibrated with CtR as primary objective and CtB band as structural control.',
            'Compare db_mean, dr_mean, turnover and sharpe against ERC/GMV/EW/CTB_ONLY.',
        ],
        'metric_deltas_vs_erc': {
            'sharpe': float(summary.loc[summary['strategy'] == 'RB_CTB_BAND', 'sharpe'].iloc[0] - summary.loc[summary['strategy'] == 'ERC', 'sharpe'].iloc[0]),
            'ann_return': float(summary.loc[summary['strategy'] == 'RB_CTB_BAND', 'ann_return'].iloc[0] - summary.loc[summary['strategy'] == 'ERC', 'ann_return'].iloc[0]),
            'ann_vol': float(summary.loc[summary['strategy'] == 'RB_CTB_BAND', 'ann_vol'].iloc[0] - summary.loc[summary['strategy'] == 'ERC', 'ann_vol'].iloc[0]),
        },
        'db_reduction_vs_erc': float(erc_db - prop_db),
        'dr_change_vs_erc': float(prop_dr - erc_dr),
        'active_rate': float(summary.loc[summary['strategy'] == 'RB_CTB_BAND', 'active_rate'].iloc[0]),
        'recommended_figures': [
            'fig_01_capital_vs_risk', 'fig_02_dr_db_frontier', 'fig_03_cumulative_nav', 'fig_04_drawdown',
            'fig_05_ctr_heatmap', 'fig_06_ctb_heatmap', 'fig_07_dr_db_timeseries', 'fig_08_corr_heatmap',
            'fig_09_calibration_heatmap', 'fig_10_solver_convergence',
        ],
        'warnings': diagnostics.get('failed_dates', {}),
        'solver_health_summary': diagnostics.get('strategy_health', {}),
        'pointers': {'figure_files': [], 'table_files': []},
        'final_parameters': {'delta': delta_star, 'eta': eta_star, 'gamma': gamma_star, 'rho': rho, 'eps_db': eps_db},
        'candidate_set_results': candidate_logs,
    }
    save_json(summaries_dir / 'analysis_pack.json', analysis_pack)
    written_files.append(str(summaries_dir / 'analysis_pack.json'))

    run_manifest = {
        'run_id': run_dir.name,
        'created_utc': pd.Timestamp.utcnow().isoformat(),
        'spec_version': 'external_freeze_0',
        'data_contract_version': 'external_freeze_1',
        'calibration_protocol_version': 'external_freeze_2',
        'code_commit': cfg['run'].get('code_commit', 'NA'),
        'universe': tickers,
        'date_range': {'start': cfg['data']['start'], 'end': cfg['data']['end']},
        'frequency': {'data': 'daily', 'rebalance': 'monthly'},
        'window_L': L,
        'covariance_method': V_method,
        'constraints': {'x_max': x_max},
        'costs': {'c': cost_c},
        'parameters': {'delta': delta_star, 'eta': eta_star, 'gamma': gamma_star, 'rho': rho, 'eps_db': eps_db},
        'outputs': written_files,
        'data_artifacts': data_artifacts,
    }
    save_json(summaries_dir / 'run_manifest.json', run_manifest)
    written_files.append(str(summaries_dir / 'run_manifest.json'))

    return {
        'split': split,
        'final_parameters': {'delta': delta_star, 'eta': eta_star, 'gamma': gamma_star, 'rho': rho, 'eps_db': eps_db},
        'written_files': written_files,
    }