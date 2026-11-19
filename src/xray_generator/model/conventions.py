"""Explicit numerical conventions shared by the new forward-model path."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np


def frozen_array(value, *, dtype=np.float64):
    """Copy input into an immutable-buffer array (including against setflags)."""
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def canonical_hash(value: Mapping | list | tuple) -> str:
    """Hash JSON-compatible semantic records; NaN is never a valid identity."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def points_mm(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("points_sample_mm must be a finite [N,3] array")
    return array


def bounds_mm(value) -> tuple[tuple[float, float], ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 2) or not np.isfinite(array).all() or np.any(array[:, 1] <= array[:, 0]):
        raise ValueError("bounds_mm must have shape [3,2] with finite increasing pairs")
    return tuple(tuple(float(v) for v in axis) for axis in array)


def q_A_inv_from_G_nm_inv(value):
    """Convert crystallographic cycles/nm to scattering-vector radians/Å."""
    return np.asarray(value, dtype=np.float64) * (2.0 * np.pi / 10.0)


def wavelength_A_from_energy_keV(value):
    energy = np.asarray(value, dtype=np.float64)
    if not np.isfinite(energy).all() or np.any(energy <= 0):
        raise ValueError("energy_keV must be finite and positive")
    return 12.398419843320026 / energy
