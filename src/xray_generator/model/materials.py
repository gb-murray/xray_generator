"""Minimal material amount records; no crystallographic intensity claim is made."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PhaseDefinition:
    """Mass-composition/density reference for a binary mixture field.

    Crystallographic bases and scattering factors are supplied by the diffraction backend.
    These records contain composition and amount information, not a transport model.
    """

    phase_id: str
    elements: tuple[str, ...]
    elemental_mass_fractions: tuple[float, ...]
    density_g_cm3: float
    source_id: str = "synthetic_fixture"

    def __post_init__(self):
        object.__setattr__(self, "density_g_cm3", float(self.density_g_cm3))
        object.__setattr__(self, "elements", tuple(self.elements))
        object.__setattr__(self, "elemental_mass_fractions", tuple(float(v) for v in self.elemental_mass_fractions))
        if not isinstance(self.phase_id,str) or not self.phase_id or not isinstance(self.source_id,str) or not self.source_id:
            raise ValueError("phase_id and source_id must be nonempty")
        if not self.elements or len(self.elements) != len(set(self.elements)) or any(not isinstance(e, str) or not e for e in self.elements):
            raise ValueError("elements must be distinct nonempty labels")
        if len(self.elements) != len(self.elemental_mass_fractions):
            raise ValueError("element labels and mass fractions must match")
        if any(not math.isfinite(v) or v < 0 for v in self.elemental_mass_fractions):
            raise ValueError("elemental_mass_fractions must be finite and nonnegative")
        if not math.isclose(sum(self.elemental_mass_fractions), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("elemental mass fractions must sum to one")
        if not math.isfinite(self.density_g_cm3) or self.density_g_cm3 <= 0:
            raise ValueError("density_g_cm3 must be finite and positive")

    def to_dict(self):
        return {"phase_id": self.phase_id, "elements": list(self.elements),
                "elemental_mass_fractions": list(self.elemental_mass_fractions),
                "density_g_cm3": self.density_g_cm3, "source_id": self.source_id}

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {"phase_id", "elements", "elemental_mass_fractions", "density_g_cm3", "source_id"}
        if unknown:
            raise ValueError(f"Unknown phase fields: {sorted(unknown)}")
        return cls(**value)
