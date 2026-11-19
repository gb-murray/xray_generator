"""Synthetic reference-rate fixture, deliberately separate from physical XRD/XRF.

The component weights are an explicitly chosen linear measurement law. They are
not a general physical law for mixing mass fractions or measured alloy spectra.
No efficiency, background or PSF is appended to already detected template rates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import readonly_array


@dataclass(frozen=True)
class ReferenceRateBackend:
    component_rates: np.ndarray
    component_ids: tuple[str, ...]
    modality: str = "xrd"
    background_included: bool = True
    profile: str = "synthetic_reference_fixture_v1"

    def __post_init__(self):
        rates = readonly_array(self.component_rates)
        ids = tuple(self.component_ids)
        if self.profile != "synthetic_reference_fixture_v1":
            raise ValueError("This backend accepts only the synthetic reference fixture profile.")
        if self.modality not in {"xrd", "xrf"}:
            raise ValueError("modality must be xrd or xrf.")
        if rates.ndim != (3 if self.modality == "xrd" else 2):
            raise ValueError("Native rates require component,row,col or component,energy axes.")
        if rates.shape[0] != len(ids) or len(set(ids)) != len(ids) or not ids:
            raise ValueError("Reference component IDs must be unique and match the rate array.")
        if rates.size == 0 or np.prod(rates.shape[1:]) > 262144:
            raise ValueError("Fixture native layout is empty or exceeds the supported channel limit.")
        object.__setattr__(self, "component_rates", rates)
        object.__setattr__(self, "component_ids", ids)

    @property
    def native_shape(self):
        return self.component_rates.shape[1:]

    @property
    def fingerprint(self):
        from ..experiment.provenance import fingerprint
        return fingerprint({"backend": "synthetic_reference_rates/1", "rates": self.component_rates,
                            "ids": self.component_ids, "modality": self.modality,
                            "background_included": self.background_included, "profile": self.profile})

    def local_rate(self, material_batch):
        if tuple(material_batch.phase_ids)!=self.component_ids:
            raise ValueError("Material phase order differs from reference components.")
        weights = np.asarray(material_batch.phase_mass_fractions, dtype=float)
        if weights.ndim != 2 or weights.shape[1] != len(self.component_ids):
            raise ValueError("Material and reference components differ.")
        occupied = np.asarray(material_batch.occupied, dtype=bool)
        rates = weights @ self.component_rates.reshape(len(self.component_ids), -1)
        rates[~occupied] = 0.0
        return rates.reshape((weights.shape[0],) + self.native_shape)

    def integration_weights(self,material_batch):
        if tuple(material_batch.phase_ids)!=self.component_ids:
            raise ValueError("Material phase order differs from reference components.")
        return np.asarray(material_batch.phase_mass_fractions)*material_batch.occupied[:,None]

    def basis_rates(self): return self.component_rates

    def basis_errors(self): return np.zeros_like(self.component_rates)


def synthetic_ring_templates(size=16):
    """Two positive detected-rate templates, not real phase fingerprints."""
    if isinstance(size, bool) or int(size) != size or not 8 <= size <= 512:
        raise ValueError("Fixture image size must be an integer between 8 and 512.")
    yy, xx = np.indices((size, size), dtype=float)
    radius = np.hypot(xx - (size-1)/2, yy - (size-1)/2)
    first = np.exp(-0.5*((radius-size*.22)/(size*.055))**2) + .03
    second = np.exp(-0.5*((radius-size*.36)/(size*.055))**2) + .03
    return np.stack([first/first.sum()*10000, second/second.sum()*14000])
