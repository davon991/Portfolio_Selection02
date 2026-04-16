from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ensure_dir, save_json


def _download_yfinance_adj_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
        group_by='column',
    )
    if data.empty:
        raise RuntimeError('Downloaded empty data from yfinance.')

    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            close = data['Adj Close'].copy()
        elif 'Close' in data.columns.get_level_values(0):
            close = data['Close'].copy()
        else:
            raise RuntimeError('Could not find Adj Close or Close in yfinance output.')
    else:
        # single ticker fallback
        col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        close = data[[col]].copy()
        close.columns = tickers[:1]

    close = close.sort_index()
    close.index = pd.to_datetime(close.index)
    return close[tickers]


def _load_csv_folder_adj_close(folder: str | Path, tickers: list[str]) -> pd.DataFrame:
    folder = Path(folder)
    frames: list[pd.DataFrame] = []
    for t in tickers:
        path = folder / f'{t}.csv'
        if not path.exists():
            raise FileNotFoundError(f'Missing CSV for ticker: {t} at {path}')
        df = pd.read_csv(path)
        lower = {c.lower(): c for c in df.columns}
        date_col = lower.get('date')
        adj_col = lower.get('adj_close') or lower.get('adj close') or lower.get('close')
        if date_col is None or adj_col is None:
            raise ValueError(f'{path} must contain date and adj_close/close columns.')
        sub = df[[date_col, adj_col]].copy()
        sub.columns = ['date', t]
        sub['date'] = pd.to_datetime(sub['date'])
        sub = sub.sort_values('date').set_index('date')
        frames.append(sub)
    out = pd.concat(frames, axis=1).sort_index()
    return out[tickers]


def _align_intersection_calendar(close: pd.DataFrame) -> pd.DataFrame:
    clean = close.dropna(how='any').copy()
    if clean.empty:
        raise RuntimeError('Intersection calendar is empty after dropping missing observations.')
    return clean


def _compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().dropna(how='any').copy()
    if rets.empty:
        raise RuntimeError('No returns available after pct_change/dropna.')
    return rets


def prepare_returns(cfg: dict[str, Any]) -> dict[str, str]:
    tickers = [str(t).upper() for t in cfg['data']['tickers']]
    start = cfg['data']['start']
    end = cfg['data']['end']
    source = cfg['data']['source']
    csv_folder = cfg['data'].get('csv_folder', 'data/raw')

    processed_dir = ensure_dir(Path('data/processed'))

    if source == 'yfinance':
        close = _download_yfinance_adj_close(tickers, start, end)
    elif source == 'csv_folder':
        close = _load_csv_folder_adj_close(csv_folder, tickers)
    else:
        raise ValueError('data.source must be one of: yfinance, csv_folder')

    aligned = _align_intersection_calendar(close)
    rets = _compute_simple_returns(aligned)

    prices_out = aligned.reset_index().rename(columns={'index': 'date'})
    prices_path = processed_dir / 'prices_daily_adj.parquet'
    prices_csv_path = processed_dir / 'prices_daily_adj.csv'
    prices_out.to_csv(prices_csv_path, index=False)
    try:
        prices_out.to_parquet(prices_path, index=False)
        prices_parquet_ok = True
    except Exception:
        prices_parquet_ok = False

    calendar = pd.DataFrame({'date': aligned.index.astype('datetime64[ns]')})
    calendar_path = processed_dir / 'calendar_master.csv'
    calendar.to_csv(calendar_path, index=False)

    returns_wide = rets.copy()
    returns_wide.index.name = 'date'
    returns_csv = processed_dir / 'returns_daily.csv'
    returns_wide.to_csv(returns_csv)

    ret_long = returns_wide.reset_index().melt(id_vars=['date'], var_name='asset', value_name='ret')
    returns_parquet = processed_dir / 'returns.parquet'
    returns_long_csv = processed_dir / 'returns_long.csv'
    ret_long.to_csv(returns_long_csv, index=False)
    try:
        ret_long.to_parquet(returns_parquet, index=False)
        returns_parquet_ok = True
    except Exception:
        returns_parquet_ok = False

    universe_mask = ret_long[['date', 'asset']].copy()
    universe_mask['is_live'] = True
    universe_mask['is_tradeable'] = True
    universe_mask['has_full_history_window'] = True
    universe_mask_path = processed_dir / 'universe_mask.csv'
    universe_mask.to_csv(universe_mask_path, index=False)

    summary_rows = []
    meta = {}
    for t in tickers:
        s = close[t]
        meta[t] = {
            'start': str(pd.to_datetime(s.first_valid_index()).date()) if s.first_valid_index() is not None else None,
            'end': str(pd.to_datetime(s.last_valid_index()).date()) if s.last_valid_index() is not None else None,
            'missing_rate_raw': float(s.isna().mean()),
            'count_raw': int(s.notna().sum()),
            'count_aligned': int(aligned[t].notna().sum()),
        }
        summary_rows.append({'asset': t, **meta[t]})
    asset_summary = pd.DataFrame(summary_rows)
    asset_summary_path = processed_dir / 'asset_summary.csv'
    asset_summary.to_csv(asset_summary_path, index=False)

    audit = {
        'source': source,
        'tickers': tickers,
        'n_assets': len(tickers),
        'start_requested': start,
        'end_requested': end,
        'aligned_start': str(aligned.index.min().date()),
        'aligned_end': str(aligned.index.max().date()),
        'n_price_rows': int(aligned.shape[0]),
        'n_return_rows': int(rets.shape[0]),
    }
    audit_path = processed_dir / 'data_audit_report.json'
    save_json(audit_path, audit)

    assets_json = processed_dir / 'assets.json'
    save_json(assets_json, meta)

    return {
        'prices_adj_parquet': str(prices_path),
        'prices_adj_csv': str(prices_csv_path),
        'prices_adj_parquet_ok': prices_parquet_ok,
        'returns_daily_csv': str(returns_csv),
        'returns_parquet': str(returns_parquet),
        'returns_long_csv': str(returns_long_csv),
        'returns_parquet_ok': returns_parquet_ok,
        'calendar_master_csv': str(calendar_path),
        'universe_mask_csv': str(universe_mask_path),
        'asset_summary_csv': str(asset_summary_path),
        'data_audit_report_json': str(audit_path),
        'assets_json': str(assets_json),
    }