"""Small public records at the native expected-event boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


def readonly_array(value, *, nonnegative=True):
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)) or (nonnegative and np.any(array < 0)):
        raise ValueError("Scientific arrays must be finite and nonnegative.")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True)
class NativeRate:
    """Events/s/native channel after linear response, before exposure/readout."""

    total_rate: np.ndarray
    numerical_error: np.ndarray
    scale_origin: str = "synthetic_reference_fixture"

    def __post_init__(self):
        values = readonly_array(self.total_rate)
        errors = readonly_array(self.numerical_error)
        if values.shape != errors.shape:
            raise ValueError("Rate and error layouts differ.")
        object.__setattr__(self, "total_rate", values)
        object.__setattr__(self, "numerical_error", errors)


class PreparedRateSolver(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def evaluate(self, snapshot, accuracy) -> NativeRate: ...
