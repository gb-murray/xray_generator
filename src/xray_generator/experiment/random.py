"""Order-independent, explicitly namespaced physical and numerical RNGs."""
from __future__ import annotations

import hashlib
import numpy as np

from .provenance import canonical_json

RNG_VERSION = "sha256_seedsequence_pcg64_v1"


def keyed_rng(root_seed: int, namespace: str, *identities) -> np.random.Generator:
    if isinstance(root_seed, bool) or not isinstance(root_seed, (int, np.integer)) or root_seed < 0:
        raise ValueError("root_seed must be a nonnegative integer")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("an explicit nonempty RNG namespace is required")
    payload = canonical_json({"version": RNG_VERSION, "root_seed": int(root_seed),
                              "namespace": namespace, "identities": identities})
    words = np.frombuffer(hashlib.sha256(payload.encode("utf-8")).digest(), dtype="<u4")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words.tolist())))
