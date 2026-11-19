"""Whole-native-pixel partitioning with exact integer sums and coverage."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _readonly(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class PolarObservation:
    counts: np.ndarray
    normalization: np.ndarray
    coverage: np.ndarray


def native_to_polar_bins(q, chi, q_edges, chi_edges):
    """Map pixel centers to one flattened q-major bin, or -1 outside q ROI.

    The full 2pi chi domain is periodic. q's last edge belongs to its last bin.
    No pixel's observed counts are fractionally split.
    """
    q, chi = np.asarray(q, dtype=float), np.asarray(chi, dtype=float)
    qe, ce = np.asarray(q_edges, dtype=float), np.asarray(chi_edges, dtype=float)
    if q.shape != chi.shape or not np.all(np.isfinite(q)) or not np.all(np.isfinite(chi)):
        raise ValueError("q/chi must be finite arrays with equal shape")
    for edges in (qe, ce):
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError("bin edges must be finite, 1D and strictly increasing")
    period = ce[-1] - ce[0]
    if not np.isclose(period, 2 * np.pi, rtol=0, atol=1e-12):
        raise ValueError("chi edges must cover a full 2pi period")
    angular = (chi - ce[0]) % period + ce[0]
    qi = np.searchsorted(qe, q, side="right") - 1
    qi = np.where(q == qe[-1], len(qe) - 2, qi)
    ci = np.minimum(np.searchsorted(ce, angular, side="right") - 1, len(ce) - 2)
    valid = (q >= qe[0]) & (q <= qe[-1])
    result = np.where(valid, qi * (len(ce) - 1) + ci, -1).astype(np.int64)
    return _readonly(result)


def partition_counts(counts, bin_indices, n_bins, *, normalizers=None, valid_mask=None):
    raw, indices = np.asarray(counts), np.asarray(bin_indices)
    if raw.dtype.kind not in "iu" or np.any(raw < 0):
        raise ValueError("native counts must be nonnegative integers")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if indices.dtype.kind not in "iu" or indices.shape != raw.shape or np.any(indices < -1) or np.any(indices >= n_bins):
        raise ValueError("bin indices must match counts and lie in [-1,n_bins)")
    if np.any(raw > np.iinfo(np.int64).max):
        raise OverflowError("native counts exceed int64 range")
    normal = np.ones(raw.shape, dtype=np.float64) if normalizers is None else np.asarray(normalizers, dtype=np.float64)
    if normal.shape != raw.shape or not np.all(np.isfinite(normal)) or np.any(normal < 0):
        raise ValueError("normalizers must be finite, nonnegative and match counts")
    valid = np.ones(raw.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask)
    if valid.shape != raw.shape or valid.dtype != np.dtype(bool):
        raise ValueError("valid_mask must be boolean and match counts")
    observed = valid & (indices >= 0)
    if int(np.sum(raw[observed], dtype=object)) > np.iinfo(np.int64).max:
        raise OverflowError("partition count sum exceeds int64 range")
    result = np.zeros(n_bins, dtype=np.int64)
    np.add.at(result, indices[observed], raw[observed].astype(np.int64))
    norm = np.zeros(n_bins, dtype=float)
    full = np.zeros(n_bins, dtype=float)
    np.add.at(norm, indices[observed], normal[observed])
    eligible = indices >= 0
    np.add.at(full, indices[eligible], normal[eligible])
    coverage = np.divide(norm, full, out=np.zeros_like(norm), where=full > 0)
    return PolarObservation(_readonly(result), _readonly(norm), _readonly(coverage))
