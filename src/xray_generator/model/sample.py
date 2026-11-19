"""Small immutable recipes for dense, extruded binary-mixture films."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .conventions import bounds_mm as validate_bounds_mm, canonical_hash, frozen_array
from .materials import PhaseDefinition


@dataclass(frozen=True)
class FractionRegion:
    """Piecewise-linear first-phase *mass* fraction, with explicit region edges."""

    x_nodes_mm: tuple[float, ...]
    first_phase_mass_fractions: tuple[float, ...]

    def __post_init__(self):
        x = tuple(float(v) for v in self.x_nodes_mm)
        y = tuple(float(v) for v in self.first_phase_mass_fractions)
        object.__setattr__(self, "x_nodes_mm", x)
        object.__setattr__(self, "first_phase_mass_fractions", y)
        if len(x) < 2 or len(x) != len(y) or not all(math.isfinite(v) for v in x + y):
            raise ValueError("A region needs matching finite node/fraction sequences of length >=2")
        if any(b <= a for a, b in zip(x, x[1:])):
            raise ValueError("x_nodes_mm must be strictly increasing")
        if any(v < 0 or v > 1 for v in y):
            raise ValueError("phase mass fractions must lie in [0,1]")

    def to_dict(self):
        return {"x_nodes_mm": list(self.x_nodes_mm),
                "first_phase_mass_fractions": list(self.first_phase_mass_fractions)}

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"x_nodes_mm", "first_phase_mass_fractions"}:
            raise ValueError("Unknown or missing fraction-region fields")
        return cls(**value)


@dataclass(frozen=True)
class MaterialRecipe:
    """A bounded static film varying along sample x, without a dense raster.

    Regions partition x; their shared edges can carry discontinuities. Values at
    an internal interface use the region to its right. Geometric support is closed
    for point queries; support-intersection calculations use positive volume.
    """

    bounds_mm: tuple[tuple[float, float], ...]
    phases: tuple[PhaseDefinition, PhaseDefinition]
    regions: tuple[FractionRegion, ...]
    realization_id: str = "material-000001"
    generator_version: str = "piecewise_linear_binary_v1"
    schema_version: str = "xray_generator.material_recipe/1.0"

    def __post_init__(self):
        object.__setattr__(self, "bounds_mm", validate_bounds_mm(self.bounds_mm))
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "regions", tuple(self.regions))
        if self.schema_version != "xray_generator.material_recipe/1.0" or self.generator_version != "piecewise_linear_binary_v1":
            raise ValueError("Unsupported material schema or generator version")
        if not isinstance(self.realization_id,str) or not self.realization_id:
            raise ValueError("realization_id must be nonempty")
        if len(self.phases) != 2 or any(not isinstance(p, PhaseDefinition) for p in self.phases):
            raise ValueError("This material provider supports exactly two PhaseDefinition records")
        if len({p.phase_id for p in self.phases}) != 2:
            raise ValueError("phase IDs must be distinct")
        if not self.regions or any(not isinstance(r, FractionRegion) for r in self.regions):
            raise ValueError("At least one FractionRegion is required")
        if self.regions[0].x_nodes_mm[0] != self.bounds_mm[0][0] or self.regions[-1].x_nodes_mm[-1] != self.bounds_mm[0][1]:
            raise ValueError("Regions must cover the complete sample x range")
        if any(a.x_nodes_mm[-1] != b.x_nodes_mm[0] for a, b in zip(self.regions, self.regions[1:])):
            raise ValueError("Regions must be ordered, adjacent and nonoverlapping")

    @property
    def fingerprint(self):
        return canonical_hash(self.to_dict())

    @classmethod
    def sharp_interface(cls, *, bounds_mm, phases, interface_x_mm=0.0,
                        left_fraction=1.0, right_fraction=0.0,
                        realization_id="material-000001"):
        limits = validate_bounds_mm(bounds_mm)
        interface = float(interface_x_mm)
        if not limits[0][0] < interface < limits[0][1]:
            raise ValueError("Interface must be inside the sample")
        return cls(limits, tuple(phases), (
            FractionRegion((limits[0][0], interface), (left_fraction, left_fraction)),
            FractionRegion((interface, limits[0][1]), (right_fraction, right_fraction)),
        ), realization_id)

    def to_dict(self):
        return {"schema_version": self.schema_version, "generator_version": self.generator_version,
                "realization_id": self.realization_id, "bounds_mm": [list(v) for v in self.bounds_mm],
                "phases": [p.to_dict() for p in self.phases], "regions": [r.to_dict() for r in self.regions]}

    @classmethod
    def from_dict(cls, value):
        allowed = {"bounds_mm", "phases", "regions", "realization_id", "generator_version", "schema_version"}
        if set(value) - allowed:
            raise ValueError(f"Unknown recipe fields: {sorted(set(value) - allowed)}")
        fields = dict(value)
        fields["phases"] = tuple(PhaseDefinition.from_dict(p) for p in fields["phases"])
        fields["regions"] = tuple(FractionRegion.from_dict(r) for r in fields["regions"])
        return cls(**fields)


@dataclass(frozen=True)
class MaterialStateBatch:
    phase_ids: tuple[str, str]
    elements: tuple[str, ...]
    occupied: np.ndarray
    phase_mass_fractions: np.ndarray
    phase_volume_fractions: np.ndarray
    elemental_mass_fractions: np.ndarray
    density_g_cm3: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "phase_ids", tuple(self.phase_ids))
        object.__setattr__(self, "elements", tuple(self.elements))
        occupied = np.asarray(self.occupied)
        if occupied.ndim != 1 or occupied.dtype.kind != "b":
            raise ValueError("occupied must be a boolean [N] array")
        if len(self.phase_ids) != 2 or len(set(self.phase_ids)) != 2 or not self.elements:
            raise ValueError("Material state requires two distinct phases and element labels")
        shapes = {"phase_mass_fractions": (len(occupied),2),
                  "phase_volume_fractions": (len(occupied),2),
                  "elemental_mass_fractions": (len(occupied),len(self.elements)),
                  "density_g_cm3": (len(occupied),)}
        for name, shape in shapes.items():
            value = np.asarray(getattr(self,name))
            if value.shape != shape or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError(f"Invalid material state array: {name}")
            if np.any(value[~occupied] != 0):
                raise ValueError("Unoccupied material state must have zero amount/density")
            if name != "density_g_cm3" and not np.allclose(value[occupied].sum(axis=1),1.,rtol=0,atol=1e-12):
                raise ValueError("Occupied material fractions must close to one")
        if np.any(np.asarray(self.density_g_cm3)[occupied] <= 0):
            raise ValueError("Occupied material density must be positive")
        object.__setattr__(self, "occupied", frozen_array(self.occupied, dtype=np.bool_))
        for name in ("phase_mass_fractions", "phase_volume_fractions", "elemental_mass_fractions", "density_g_cm3"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
