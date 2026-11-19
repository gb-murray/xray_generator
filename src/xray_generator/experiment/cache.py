"""Byte-bounded clean expectation caches with atomic, verified disk entries.

The error array is a numerical certificate supplied by the solver. This module
checks its provenance/integrity and requested tolerance; it does not turn an
estimated quadrature error into a proven physical bound.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
import zipfile

import numpy as np

from .provenance import canonical_json, expectation_key, filesystem_path


def _readonly(value):
    array = np.asarray(value, dtype=np.float64)
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.float64).reshape(array.shape)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class CachedExpectation:
    values: np.ndarray
    error: np.ndarray
    metadata: Mapping

    @property
    def nbytes(self):
        return self.values.nbytes + self.error.nbytes + len(canonical_json(_thaw(self.metadata)).encode("utf-8")) + 64


class ExpectationCache:
    """LRU memory plus optional content-addressed disk cache.

    ``max_memory_bytes=0`` disables memory storage. Entries too large for a
    level bypass it without affecting the computed expectation. Disk files are
    compressed but their uncompressed array size must also fit the disk cap.
    Cache reads never return an entry with a looser requested error certificate.
    """
    def __init__(self, max_memory_bytes=128 * 1024**2, *, disk_directory=None,
                 max_disk_bytes=0):
        for name, value in (("max_memory_bytes", max_memory_bytes), ("max_disk_bytes", max_disk_bytes)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.max_memory_bytes = int(max_memory_bytes)
        self.max_disk_bytes = int(max_disk_bytes)
        self.disk_directory = filesystem_path(disk_directory) if disk_directory is not None else None
        if self.disk_directory is not None and self.max_disk_bytes:
            self.disk_directory.mkdir(parents=True, exist_ok=True)
        self._entries = OrderedDict()
        self._bytes_used = 0
        self.hits = self.misses = self.disk_hits = self.evictions = self.invalidations = 0

    @property
    def bytes_used(self):
        return self._bytes_used

    @property
    def disk_bytes_used(self):
        total = 0
        for path in self._disk_paths():
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                pass  # another worker evicted this regenerable entry
        return total

    def _disk_paths(self):
        if self.disk_directory is None or not self.disk_directory.exists():
            return []
        return [path for path in self.disk_directory.glob("expectation-*.npz")
                if path.is_file() and re.fullmatch(r"expectation-[0-9a-f]{64}\.npz", path.name)]

    def _path(self, key):
        return self.disk_directory / f"expectation-{key}.npz"

    @staticmethod
    def _check_key(key):
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("cache key must be a lowercase SHA-256 dependency fingerprint")

    @staticmethod
    def _entry(values, error, metadata):
        values = _readonly(values)
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("expected events must be nonempty, finite and nonnegative")
        raw_error = np.asarray(error, dtype=np.float64)
        try:
            error = _readonly(np.broadcast_to(raw_error, values.shape))
        except ValueError as exc:
            raise ValueError("error certificate is not broadcastable to expected events") from exc
        if not np.all(np.isfinite(error)) or np.any(error < 0):
            raise ValueError("error certificate must be finite and nonnegative")
        normalized = json.loads(canonical_json(metadata or {}))
        if not isinstance(normalized, dict):
            raise TypeError("cache metadata must be a mapping")
        return CachedExpectation(values, error, _freeze(normalized))

    def _remember(self, key, entry):
        if key in self._entries:
            self._bytes_used -= self._entries.pop(key).nbytes
        if entry.nbytes > self.max_memory_bytes:
            return False
        while self._entries and self._bytes_used + entry.nbytes > self.max_memory_bytes:
            _, old = self._entries.popitem(last=False)
            self._bytes_used -= old.nbytes
            self.evictions += 1
        self._entries[key] = entry
        self._bytes_used += entry.nbytes
        return True

    @staticmethod
    def _payload(key, entry):
        header = {"format": "native_expectation_cache_v1", "key": key,
                  "shape": list(entry.values.shape), "metadata": _thaw(entry.metadata)}
        header_text = canonical_json(header)
        digest = hashlib.sha256(header_text.encode("utf-8") + entry.values.tobytes() + entry.error.tobytes()).hexdigest()
        return header_text, digest

    def put(self, key, values, *, error=0., metadata=None):
        self._check_key(key)
        entry = self._entry(values, error, metadata)
        remembered = self._remember(key, entry)
        written = False
        if self.disk_directory is not None and self.max_disk_bytes and entry.nbytes <= self.max_disk_bytes:
            header, digest = self._payload(key, entry)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(prefix=".expectation-", suffix=".partial",
                                                 dir=self.disk_directory, delete=False) as handle:
                    temporary = Path(handle.name)
                    np.savez_compressed(handle, values=entry.values, error=entry.error,
                                        header=np.frombuffer(header.encode("utf-8"), dtype=np.uint8),
                                        checksum=np.frombuffer(digest.encode("ascii"), dtype=np.uint8))
                    handle.flush()
                    os.fsync(handle.fileno())
                if temporary.stat().st_size <= self.max_disk_bytes:
                    os.replace(temporary, self._path(key))
                    written = True
                    self._prune_disk()
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        return remembered or written

    def _prune_disk(self):
        records = []
        for path in self._disk_paths():
            try:
                stat = path.stat()
                records.append((stat.st_mtime_ns, path.name, path, stat.st_size))
            except FileNotFoundError:
                pass
        records.sort()
        total = sum(record[3] for record in records)
        for _, _, path, size in records:
            if total <= self.max_disk_bytes:
                break
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            total -= size

    def _load(self, key):
        if self.disk_directory is None or not self.max_disk_bytes:
            return None
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            if path.stat().st_size > self.max_disk_bytes:
                raise ValueError("disk entry exceeds cache cap")
            with zipfile.ZipFile(path) as archive:
                if sum(item.file_size for item in archive.infolist()) > self.max_disk_bytes + 4096:
                    raise ValueError("uncompressed cache entry exceeds cap")
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"values", "error", "header", "checksum"}:
                    raise ValueError("unknown cache entry fields")
                header_text = data["header"].tobytes().decode("utf-8")
                header = json.loads(header_text)
                if header["format"] != "native_expectation_cache_v1" or header["key"] != key:
                    raise ValueError("cache format/key mismatch")
                if data["values"].dtype != np.dtype("float64") or data["error"].dtype != np.dtype("float64"):
                    raise ValueError("cache dtype mismatch")
                entry = self._entry(data["values"], data["error"], header["metadata"])
                canonical_header, checksum = self._payload(key, entry)
                if canonical_header != header_text or data["checksum"].tobytes().decode("ascii") != checksum:
                    raise ValueError("cache content checksum mismatch")
                if list(entry.values.shape) != header["shape"] or entry.nbytes > self.max_disk_bytes:
                    raise ValueError("cache shape or size mismatch")
            os.utime(path, None)
            return entry
        except (OSError, ValueError, TypeError, KeyError, UnicodeError, zipfile.BadZipFile, EOFError):
            self.invalidations += 1
            return None

    def get(self, key, *, max_error=None):
        self._check_key(key)
        entry = self._entries.get(key)
        disk = False
        if entry is None:
            entry = self._load(key)
            disk = entry is not None
        if entry is None:
            self.misses += 1
            return None
        if max_error is not None:
            required = np.asarray(max_error, dtype=np.float64)
            if not np.all(np.isfinite(required)) or np.any(required < 0):
                raise ValueError("requested error allowance must be finite and nonnegative")
            try:
                acceptable = np.all(entry.error <= np.broadcast_to(required, entry.values.shape))
            except ValueError as exc:
                raise ValueError("requested error shape is incompatible") from exc
            if not acceptable:
                self.misses += 1
                return None
        self._remember(key, entry)
        self._entries.move_to_end(key) if key in self._entries else None
        self.hits += 1
        self.disk_hits += int(disk)
        return entry

    def clear_memory(self):
        self._entries.clear()
        self._bytes_used = 0

    def info(self):
        return {"hits": self.hits, "misses": self.misses, "disk_hits": self.disk_hits,
                "evictions": self.evictions, "invalidations": self.invalidations,
                "memory_bytes": self.bytes_used, "disk_bytes": self.disk_bytes_used}
