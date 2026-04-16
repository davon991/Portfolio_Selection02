from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils import capped_simplex_project


@dataclass
class SolverResult:
    x: np.ndarray
    success: bool
    message: str
    nit: int


def solve_ew(n: int, x_max: float) -> SolverResult:
    x0 = np.full(n, 1.0 / n)
    x = capped_simplex_project(x0, x_max=x_max)
    return SolverResult(x=x, success=True, message='equal_weight', nit=0)