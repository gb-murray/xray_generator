"""Low-rate native Poisson readout; analysis masks do not alter expectations."""
from __future__ import annotations

import numpy as np
from ..experiment.random import keyed_rng


def poisson_readout(expected, *, root_seed=0, readout_id="0"):
    mean = np.asarray(expected, dtype=np.float64)
    if mean.size == 0 or not np.all(np.isfinite(mean)) or np.any(mean < 0):
        raise ValueError("expected native events must be finite and nonnegative")
    maximum = np.iinfo(np.int64).max
    if np.any(mean > maximum - 10 * np.sqrt(float(maximum))):
        raise OverflowError("Poisson expectation is too large for int64 readout")
    counts = keyed_rng(root_seed, "readout", readout_id).poisson(mean)
    return np.frombuffer(np.ascontiguousarray(counts, dtype=np.int64).tobytes(), dtype=np.int64).reshape(mean.shape)


def apply_analysis_mask(counts, valid_mask):
    """Return an excluded-channel view while retaining caller-owned raw counts."""
    raw = np.asarray(counts)
    mask = np.asarray(valid_mask)
    if raw.dtype.kind not in "iu" or np.any(raw < 0):
        raise ValueError("raw native counts must be nonnegative integers")
    if mask.dtype != np.dtype(bool) or mask.shape != raw.shape:
        raise ValueError("analysis validity must be a boolean array matching counts")
    result = np.where(mask, raw, 0)
    return np.frombuffer(np.ascontiguousarray(result).tobytes(), dtype=result.dtype).reshape(result.shape)
