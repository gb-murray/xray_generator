"""Canonical content fingerprints for numerical requests and immutable assets."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


def filesystem_path(path) -> Path:
    """Resolve an OS path, using Windows' extended absolute path convention.

    Atomic rename in some Windows runtimes still enforces MAX_PATH on ordinary
    absolute paths even when Python successfully created the source file.
    Extended paths preserve the same target; they are not a different directory.
    """
    resolved = str(Path(path).resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        resolved = "\\\\?\\UNC\\" + resolved[2:] if resolved.startswith("\\\\") else "\\\\?\\" + resolved
    return Path(resolved)


def canonical_data(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: canonical_data(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays have no supported canonical fingerprint")
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError("nonfinite arrays have no supported canonical fingerprint")
        data = np.ascontiguousarray(value)
        return {"array_dtype": data.dtype.str, "array_shape": list(data.shape),
                "array_sha256": hashlib.sha256(data.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return canonical_data(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("fingerprint dictionary keys must be strings")
        return {key: canonical_data(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [canonical_data(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite floats have no supported canonical fingerprint")
        return value
    raise TypeError(f"unsupported fingerprint value: {type(value).__name__}")


def canonical_json(value) -> str:
    return json.dumps(canonical_data(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def fingerprint(value, *, namespace="xray_generator.content_v1") -> str:
    encoded = canonical_json({"namespace": namespace, "content": value}).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expectation_key(dependencies) -> str:
    """Hash the caller's complete physical/numerical dependency record.

    Readout seeds and analysis masks belong downstream. The caller must include
    source, recipe/assets, true response, trajectory, background and accuracy.
    """
    return fingerprint(dependencies, namespace="xray_generator.native_expectation_v1")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
