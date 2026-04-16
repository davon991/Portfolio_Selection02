from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_dir, save_json


def _save_fig(base: Path, formats: list[str]) -> list[str]:
    written = []
    for fmt in formats:
        out = base.with_suffix(f'.{fmt}')
        plt.tight_layout()
        plt.savefig(out, dpi=160, bbox_inches='tight')
        written.append(str(out))
    return written


def _write_tex_table(path: Path, df: pd.DataFrame, index: bool = False) -> None:
    tex = df.to_latex(index=index, float_format=lambda x: f'{x:.6f}' if isinstance(x, float) else str(x))
    path.write_text(tex, encoding='utf-8')


def make_all_figures(cfg: dict[str, Any], run_dir: Path) -> list[str]:
    run_dir = Path(run_dir)
    panels_dir = run_dir / 'panels'
    summaries_dir = run_dir / 'summaries'
    figures_dir = ensure_dir(run_dir / 'figures')
    tables_dir = ensure_dir(run_dir / 'tables')
    fmts = list(cfg.get('reporting', {}).get('figure_format', ['png', 'pdf']))
    written_all: list[str] = []

    weights = pd.read_csv(panels_dir / 'weights.csv')
    ctr = pd.read_csv(panels_dir / 'ctr_long.csv')
    ctb = pd.read_csv(panels_dir / 'ctb_long.csv')
    drdb = pd.read_csv(panels_dir / 'dr_db_timeseries.csv')
    perf = pd.read_csv(panels_dir / 'perf_daily.csv')
    summary = pd.read_csv(summaries_dir / 'summary_metrics.csv')
    cal = pd.read_csv(run_dir / 'logs' / 'calibration_log.csv')
    solver_trace = pd.read_csv(panels_dir / 'solver_trace.csv')

    weights['date'] = pd.to_datetime(weights['date'])
    ctr['date'] = pd.to_datetime(ctr['date'])
    ctb['date'] = pd.to_datetime(ctb['date'])
    drdb['date'] = pd.to_datetime(drdb['date'])
    perf['date'] = pd.to_datetime(perf['date'])

    prop = 'RB_CTB_BAND'
    last_date = weights.loc[weights['strategy'] == prop, 'date'].max()

    # fig_01 capital vs risk
    plt.figure(figsize=(10, 4))
    w_sub = weights[(weights['strategy'] == prop) & (weights['date'] == last_date)].sort_values('weight', ascending=False)
    c_sub = ctr[(ctr['strategy'] == prop) & (ctr['date'] == last_date)].rename(columns={'ctr': 'risk_share'})
    merged = w_sub.merge(c_sub[['asset', 'risk_share']], on='asset', how='left')
    x = range(len(merged))
    plt.bar(x, merged['weight'], label='Capital Weight', alpha=0.7)
    plt.bar(x, -merged['risk_share'], label='Risk Share (negative mirror)', alpha=0.7)
    plt.xticks(list(x), merged['asset'], rotation=45, ha='right')
    plt.axhline(0.0, color='black', linewidth=0.8)
    plt.title('Capital Allocation vs Risk Allocation - RB_CTB_BAND')
    plt.legend()
    written_all += _save_fig(figures_dir / 'fig_01_capital_vs_risk', fmts)
    plt.close()

    # fig_02 dr-db frontier
    plt.figure(figsize=(6, 4))
    plt.scatter(summary['db_mean'], summary['dr_mean'])
    for _, row in summary.iterrows():
        plt.annotate(row['strategy'], (row['db_mean'], row['dr_mean']))
    plt.xlabel('Mean D_B')
    plt.ylabel('Mean D_R')
    plt.title('CtR-CtB Mechanism Frontier')
    written_all += _save_fig(figures_dir / 'fig_02_dr_db_frontier', fmts)
    plt.close()

    # fig_03 cumulative nav
    plt.figure(figsize=(10, 4))
    for sname, sub in perf.groupby('strategy'):
        plt.plot(sub['date'], sub['nav'], label=sname)
    plt.legend()
    plt.title('Cumulative NAV')
    written_all += _save_fig(figures_dir / 'fig_03_cumulative_nav', fmts)
    plt.close()

    # fig_04 drawdown
    plt.figure(figsize=(10, 4))
    for sname, sub in perf.groupby('strategy'):
        plt.plot(sub['date'], sub['drawdown'], label=sname)
    plt.legend()
    plt.title('Drawdown')
    written_all += _save_fig(figures_dir / 'fig_04_drawdown', fmts)
    plt.close()

    # fig_05 ctr heatmap (pivot)
    ctr_sub = ctr[ctr['strategy'] == prop].pivot(index='asset', columns='date', values='ctr')
    plt.figure(figsize=(10, 5))
    plt.imshow(ctr_sub.values, aspect='auto')
    plt.yticks(range(len(ctr_sub.index)), ctr_sub.index)
    plt.title('CtR Heatmap - RB_CTB_BAND')
    written_all += _save_fig(figures_dir / 'fig_05_ctr_heatmap', fmts)
    plt.close()

    # fig_06 ctb heatmap
    ctb_sub = ctb[ctb['strategy'] == prop].pivot(index='asset', columns='date', values='ctb')
    plt.figure(figsize=(10, 5))
    plt.imshow(ctb_sub.values, aspect='auto')
    plt.yticks(range(len(ctb_sub.index)), ctb_sub.index)
    plt.title('CtB Heatmap - RB_CTB_BAND')
    written_all += _save_fig(figures_dir / 'fig_06_ctb_heatmap', fmts)
    plt.close()

    # fig_07 dr/db timeseries
    plt.figure(figsize=(10, 4))
    sub = drdb[drdb['strategy'] == prop]
    plt.plot(sub['date'], sub['dr'], label='D_R')
    plt.plot(sub['date'], sub['db'], label='D_B')
    if 'delta' in sub.columns and sub['delta'].notna().any():
        plt.plot(sub['date'], sub['delta'], label='delta')
    plt.legend()
    plt.title('D_R and D_B Timeseries - RB_CTB_BAND')
    written_all += _save_fig(figures_dir / 'fig_07_dr_db_timeseries', fmts)
    plt.close()

    # fig_08 corr heatmap
    corr_path = panels_dir / 'corr_structure_panel.csv'
    if corr_path.exists():
        corr_df = pd.read_csv(corr_path)
        pivot = corr_df.pivot(index='asset_i', columns='asset_j', values='corr')
        plt.figure(figsize=(7, 6))
        plt.imshow(pivot.values, aspect='auto')
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha='right')
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title('Correlation Heatmap')
        written_all += _save_fig(figures_dir / 'fig_08_corr_heatmap', fmts)
        plt.close()

    # fig_09 calibration heatmap (p vs metric)
    if not cal.empty and 'p' in cal.columns and 'score' in cal.columns:
        cal2 = cal.copy()
        cal2['gamma'] = cal2['gamma'].round(12)
        pivot = cal2.pivot_table(index='gamma', columns='p', values='score', aggfunc='mean')
        plt.figure(figsize=(7, 4))
        plt.imshow(pivot.values, aspect='auto')
        plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
        plt.yticks(range(len(pivot.index)), [f'{r:.2e}' for r in pivot.index])
        plt.title('Calibration Heatmap (score)')
        written_all += _save_fig(figures_dir / 'fig_09_calibration_heatmap', fmts)
        plt.close()

    # fig_10 solver convergence
    plt.figure(figsize=(10, 4))
    for sname, sub in solver_trace.groupby('strategy'):
        s2 = sub.sort_values('date')
        plt.plot(pd.to_datetime(s2['date']), s2['nit'], label=sname)
    plt.legend()
    plt.title('Solver Iteration Trace')
    written_all += _save_fig(figures_dir / 'fig_10_solver_convergence', fmts)
    plt.close()

    # tables
    table_01 = summary.copy()
    p = tables_dir / 'table_01_summary_metrics.csv'
    table_01.to_csv(p, index=False)
    written_all.append(str(p))
    p2 = tables_dir / 'table_01_summary_metrics.tex'
    _write_tex_table(p2, table_01)
    written_all.append(str(p2))

    mech = summary[['strategy', 'dr_mean', 'db_mean', 'active_rate', 'turnover_mean', 'turnover_p95']].copy()
    p = tables_dir / 'table_02_mechanism_metrics.csv'
    mech.to_csv(p, index=False)
    written_all.append(str(p))
    p2 = tables_dir / 'table_02_mechanism_metrics.tex'
    _write_tex_table(p2, mech)
    written_all.append(str(p2))

    rob = pd.DataFrame(cal)
    p = tables_dir / 'table_03_robustness.csv'
    rob.to_csv(p, index=False)
    written_all.append(str(p))
    p2 = tables_dir / 'table_03_robustness.tex'
    _write_tex_table(p2, rob)
    written_all.append(str(p2))

    assets_manifest = {
        'copied_figures': written_all,
        'source_files': {
            'summary_metrics': str(summaries_dir / 'summary_metrics.csv'),
            'analysis_pack': str(summaries_dir / 'analysis_pack.json'),
            'diagnostics': str(summaries_dir / 'diagnostics.json'),
        },
    }
    save_json(run_dir / 'paper_assets_manifest.json', assets_manifest)
    written_all.append(str(run_dir / 'paper_assets_manifest.json'))
    return written_all