from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, obj: Any) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'isoformat'):
        try:
            return obj.isoformat()
        except Exception:
            pass
    return str(obj)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compute_run_id(cfg: dict[str, Any]) -> str:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode('utf-8')
    short_hash = hashlib.sha256(payload).hexdigest()[:8]
    name = cfg.get('run', {}).get('name', 'run')
    return f'{stamp}-{name}-{short_hash}'


def read_version_from_frozen_doc(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return 'MISSING'
    text = p.read_text(encoding='utf-8', errors='ignore')
    patterns = [r'(?im)^version\s*[:=]\s*([\w\.-]+)', r'(?im)^版本\s*[:：]\s*([\w\.-]+)']
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return sha256_file(p)[:12]


def capped_simplex_project(v: np.ndarray, x_max: float, tol: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = v.size
    if x_max <= 0:
        raise ValueError('x_max must be positive')
    if x_max * n < 1.0 - tol:
        raise ValueError('Infeasible box-simplex: n * x_max < 1')

    lo = np.min(v) - x_max
    hi = np.max(v)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        x = np.clip(v - mid, 0.0, x_max)
        s = float(x.sum())
        if abs(s - 1.0) <= tol:
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    x = np.clip(v - 0.5 * (lo + hi), 0.0, x_max)
    s = float(x.sum())
    if s <= tol:
        x = np.full(n, 1.0 / n)
        x = np.clip(x, 0.0, x_max)
        x = x / x.sum()
        return x
    x = x / s
    x = np.clip(x, 0.0, x_max)
    x = x / x.sum()
    return x


def month_end_dates(index: Iterable) -> list:
    dates = list(index)
    if not dates:
        return []
    out = []
    prev = None
    for d in dates:
        ym = (d.year, d.month)
        if prev is not None and ym != prev[0]:
            out.append(prev[1])
        prev = (ym, d)
    if prev is not None:
        out.append(prev[1])
    return out


def longest_false_streak(flags: Iterable[bool]) -> int:
    best = 0
    cur = 0
    for flag in flags:
        if flag:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best