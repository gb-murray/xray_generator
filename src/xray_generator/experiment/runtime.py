"""Fingerprints of the loaded source tree and numerical runtime for cache keys."""
from functools import lru_cache
from pathlib import Path
import platform

import numpy as np
import scipy

from .provenance import file_sha256, fingerprint


@lru_cache(maxsize=1)
def runtime_identity():
    # One interpreter executes one imported implementation. A new process reads
    # newly installed sources; no commit ID can hide uncommitted source changes.
    package_root=Path(__file__).resolve().parents[1]
    sources={p.relative_to(package_root).as_posix():file_sha256(p)
             for p in sorted(package_root.rglob("*.py"))}
    return fingerprint({"sources":sources,"python":platform.python_version(),
                        "numpy":np.__version__,"scipy":scipy.__version__})
