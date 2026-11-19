"""Bounded, boundary-aware integration of vector-valued expectations.

The Gauss--Kronrod error is an estimate, not a proof. Known interfaces and
narrow inclusions must be supplied as breakpoints; no quadrature rule can
certify an arbitrary unobserved spike from a finite set of function values.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np


def _readonly(value):
    array = np.asarray(value, dtype=np.float64)
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True)
class IntegrationOptions:
    atol: float = 1e-8
    rtol: float = 1e-6
    max_evaluations: int = 10000
    max_intervals: int = 1024
    max_depth: int = 24
    max_working_bytes: int = 256 * 1024**2

    def __post_init__(self):
        for name in ("atol", "rtol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.atol == 0 and self.rtol == 0:
            raise ValueError("at least one integration tolerance must be positive")
        for name in ("max_evaluations", "max_intervals", "max_depth", "max_working_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class IntegrationResult:
    value: np.ndarray
    error: np.ndarray
    evaluations: int
    intervals: int
    converged: bool
    error_kind: str = "gauss_kronrod_estimate"


class IntegrationError(RuntimeError):
    def __init__(self, message, partial_result=None):
        super().__init__(message)
        self.partial_result = partial_result


# Nonnegative abscissas of Kronrod 15; odd indices plus zero form Gauss 7.
_X = np.array([.9914553711208126, .9491079123427585, .8648644233597691,
               .7415311855993945, .5860872354676911, .4058451513773972,
               .2077849550078985, 0.0])
_WK = np.array([.02293532201052922, .06309209262997855, .1047900103222502,
                .1406532597155259, .1690047266392679, .1903505780647854,
                .2044329400752989, .2094821410847278])
_WG = np.array([.1294849661688697, .2797053914892766,
                .3818300505051189, .4179591836734694])


def integrate_vector(function: Callable[[float], np.ndarray], a: float, b: float,
                     *, options: IntegrationOptions | None = None,
                     breakpoints: Iterable[float] = ()) -> IntegrationResult:
    """Integrate every output channel to ``atol + rtol * abs(value)``.

    Breakpoints are absolute coordinates strictly inside [a,b]; endpoints are
    accepted and ignored. Descending/nonfinite intervals and changing shapes
    fail. The zero-width case evaluates once to establish output shape.
    Failure to meet the requested accuracy raises ``IntegrationError``.
    """
    options = options or IntegrationOptions()
    if not isinstance(options, IntegrationOptions):
        raise TypeError("options must be IntegrationOptions")
    if not math.isfinite(a) or not math.isfinite(b) or b < a:
        raise ValueError("integration endpoints must be finite and ascending")
    points = []
    for point in breakpoints:
        if not math.isfinite(point) or point < a or point > b:
            raise ValueError("breakpoints must lie within the integration interval")
        if a < point < b:
            points.append(float(point))
    edges = [float(a), *sorted(set(points)), float(b)]
    if len(edges) - 1 > options.max_intervals:
        raise IntegrationError("initial breakpoint intervals exceed max_intervals")
    shape = None
    evaluations = 0
    vector_bytes = 0

    def check_memory(interval_count):
        # Conservative bound for panel values/errors, reductions, and GK15
        # scratch vectors. User callback storage is owned by the caller.
        if (4 * interval_count + 40) * vector_bytes > options.max_working_bytes:
            raise IntegrationError("estimated vector working set exceeds max_working_bytes")

    def evaluate(x):
        nonlocal shape, evaluations, vector_bytes
        if evaluations >= options.max_evaluations:
            raise IntegrationError("max_evaluations reached before convergence")
        evaluations += 1
        value = np.asarray(function(float(x)), dtype=np.float64)
        if shape is None:
            shape = value.shape
            vector_bytes = value.nbytes
            check_memory(max(1, len(edges) - 1))
        if value.shape != shape:
            raise ValueError("integrand output shape changed")
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("integrand must return a nonempty finite array")
        return value.copy()

    if a == b:
        zero = np.zeros_like(evaluate(a))
        return IntegrationResult(_readonly(zero), _readonly(zero), evaluations, 0, True)

    def panel(left, right, depth):
        center = left + (right - left) / 2
        half = (right - left) / 2
        if half <= 0 or center in (left, right):
            raise IntegrationError("interval cannot be subdivided at float64 precision")
        middle = evaluate(center)
        kronrod = _WK[7] * middle
        gauss = _WG[3] * middle
        absolute = _WK[7] * np.abs(middle)
        pairs = []
        for index in range(7):
            below = evaluate(center - half * _X[index])
            above = evaluate(center + half * _X[index])
            pair = below + above
            pairs.append((below, above))
            kronrod += _WK[index] * pair
            absolute += _WK[index] * (np.abs(below) + np.abs(above))
            if index % 2:
                gauss += _WG[(index - 1) // 2] * pair
        mean = kronrod / 2
        asc = _WK[7] * np.abs(middle - mean)
        for index, (below, above) in enumerate(pairs):
            asc += _WK[index] * (np.abs(below - mean) + np.abs(above - mean))
        value = kronrod * half
        absolute *= half
        asc *= half
        error = np.abs((kronrod - gauss) * half)
        ratio = np.divide(200 * error, asc, out=np.zeros_like(error), where=asc > 0)
        error = np.where((asc > 0) & (error > 0), asc * np.minimum(1., ratio**1.5), error)
        error = np.maximum(error, 50 * np.finfo(float).eps * absolute)
        return (left, right, depth, value, error)

    panels = [panel(left, right, 0) for left, right in zip(edges[:-1], edges[1:])]

    def current(converged):
        values = np.stack([p[3] for p in panels])
        errors = np.stack([p[4] for p in panels])
        return IntegrationResult(_readonly(values.sum(axis=0)), _readonly(errors.sum(axis=0)),
                                 evaluations, len(panels), converged)

    while True:
        result = current(False)
        tolerance = options.atol + options.rtol * np.abs(result.value)
        if np.all(result.error <= tolerance):
            return IntegrationResult(result.value, result.error, evaluations, len(panels), True)
        if len(panels) >= options.max_intervals:
            raise IntegrationError("max_intervals reached before convergence", result)
        if evaluations + 30 > options.max_evaluations:
            raise IntegrationError("max_evaluations reached before convergence", result)
        check_memory(len(panels) + 1)
        candidates = [i for i, p in enumerate(panels) if p[2] < options.max_depth]
        if not candidates:
            raise IntegrationError("max_depth reached before convergence", result)
        # Recompute normalized priority because the total-value tolerance may change.
        def priority(i):
            error = panels[i][4]
            ratio = np.divide(error, tolerance, out=np.full_like(error, np.inf), where=tolerance > 0)
            ratio = np.where((error == 0) & (tolerance == 0), 0, ratio)
            return float(np.max(ratio))
        index = max(candidates, key=priority)
        left, right, depth, _, _ = panels[index]
        center = left + (right - left) / 2
        children = [panel(left, center, depth + 1), panel(center, right, depth + 1)]
        panels[index:index + 1] = children
