from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_full_pipeline
from src.data_prep import prepare_returns
from src.reporting import make_all_figures
from src.utils import compute_run_id, ensure_dir, load_yaml, read_version_from_frozen_doc, save_json, sha256_file


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    results_root = ensure_dir(Path(cfg['run']['results_dir']))
    run_id = compute_run_id(cfg)
    run_dir = ensure_dir(results_root / run_id)

    # These three are external frozen docs in the user's main repository.
    spec_path = Path('paper/spec.md')
    dc_path = Path('data/data_contract.md')
    cal_path = Path('paper/calibration_protocol.md')
    spec_version = read_version_from_frozen_doc(spec_path)
    data_contract_version = read_version_from_frozen_doc(dc_path)
    calibration_protocol_version = read_version_from_frozen_doc(cal_path)

    data_artifacts = prepare_returns(cfg)
    outputs = run_full_pipeline(cfg, run_dir, data_artifacts)

    config_path = run_dir / 'config.json'
    save_json(config_path, cfg)

    figure_files = []
    if cfg.get('reporting', {}).get('make_figures', True):
        figure_files = make_all_figures(cfg, run_dir)

    analysis_pack_path = run_dir / 'summaries' / 'analysis_pack.json'
    if analysis_pack_path.exists():
        ap = _load_json(analysis_pack_path)
        ap.setdefault('pointers', {})
        ap['pointers'].setdefault('table_files', [])
        ap['pointers']['figure_files'] = figure_files
        save_json(analysis_pack_path, ap)

    written_files = list(outputs.get('written_files', [])) + figure_files + [str(config_path)]

    run_manifest = {
        'run_id': run_id,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'spec_version': spec_version,
        'data_contract_version': data_contract_version,
        'calibration_protocol_version': calibration_protocol_version,
        'code_commit': cfg['run'].get('code_commit', 'NA'),
        'python_version': sys.version.replace('\n', ' '),
        'platform': platform.platform(),
        'universe': cfg['data']['tickers'],
        'date_range': {'start': cfg['data']['start'], 'end': cfg['data']['end']},
        'frequency': {'data': 'daily', 'rebalance': 'monthly'},
        'window_L': int(cfg['experiment']['window_L']),
        'covariance_method': cfg['experiment']['covariance_method'],
        'constraints': {'x_max': float(cfg['experiment']['x_max'])},
        'costs': {'c': float(cfg['experiment']['cost_c'])},
        'parameters': {
            'delta': outputs['final_parameters']['delta'],
            'eta': outputs['final_parameters']['eta'],
            'gamma': outputs['final_parameters']['gamma'],
            'rho': float(cfg['solver']['rho']),
            'eps_db': float(cfg['experiment']['eps_db']),
        },
        'data_hashes': {
            'returns_primary': sha256_file(Path(data_artifacts['returns_parquet'])) if Path(data_artifacts['returns_parquet']).exists() else sha256_file(Path(data_artifacts['returns_long_csv'])),
            'calendar_master_csv': sha256_file(Path(data_artifacts['calendar_master_csv'])),
        },
        'outputs': written_files,
    }
    manifest_path = run_dir / 'summaries' / 'run_manifest.json'
    save_json(manifest_path, run_manifest)

    print(f'[OK] run_id = {run_id}')
    print(f'[OK] results = {run_dir}')
    print(f'[OK] manifest = {manifest_path}')


if __name__ == '__main__':
    main()