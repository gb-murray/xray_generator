"""Explicit-cell kinematic crystallography with neutral atomic form factors.

Reciprocal vectors use cycles/angstrom, not radians/angstrom. Neutral IT92 f0
is evaluated at s=|G|/2; anomalous, absorption and thermal factors are not
silently invented. The small neutral-atom subset is separately attributed in
``data/it92_subset.json`` with its upstream license.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import itertools
import json
import math

import numpy as np

from ..experiment.provenance import fingerprint
from ..model.conventions import frozen_array


_DATA_BYTES = files(__package__).joinpath("data/it92_subset.json").read_bytes()
_ATOMIC_DATA = json.loads(_DATA_BYTES)
ATOMIC_DATA_FINGERPRINT = hashlib.sha256(_DATA_BYTES).hexdigest()
# Standard atomic weights: NIST Atomic Weights and Isotopic Compositions tables.
ATOMIC_MASSES_U = {"Cu": 63.546, "Ti": 47.867, "Fe": 55.845, "Nb": 92.90637}
# Pinned CODATA 2018 atomic mass constant; nominal cell/density uncertainty is
# substantially larger than subsequent adjustments of this constant.
ATOMIC_MASS_UNIT_G = 1.66053906660e-24


def neutral_form_factor(element: str, G_norm_A_inv):
    """Unmodified neutral IT92 f0 in electrons; domain 0 <= |G| < 4 Å^-1."""
    if element not in _ATOMIC_DATA["elements"]:
        raise ValueError(f"No supported neutral scattering data for {element!r}.")
    g = np.asarray(G_norm_A_inv, dtype=np.float64)
    if not np.all(np.isfinite(g)) or np.any(g < 0) or np.any(g >= 4):
        raise ValueError("IT92 requires finite 0 <= |G| < 4 inverse angstrom.")
    coef = _ATOMIC_DATA["elements"][element]
    s2 = (g / 2) ** 2
    value = np.zeros_like(g) + coef["c"]
    for a, b in zip(coef["a"], coef["b"]):
        value += a * np.exp(-b * s2)
    return value


@dataclass(frozen=True)
class AtomicSite:
    element: str
    fractional_xyz: tuple[float, float, float]
    occupancy: float = 1.0

    def __post_init__(self):
        xyz = tuple(float(v) for v in self.fractional_xyz)
        occ = float(self.occupancy)
        if self.element not in ATOMIC_MASSES_U:
            raise ValueError("Unsupported element in atomic basis.")
        if len(xyz) != 3 or not np.all(np.isfinite(xyz)) or any(not 0 <= v < 1 for v in xyz):
            raise ValueError("Atomic fractional coordinates must be finite and in [0,1).")
        if not math.isfinite(occ) or not 0 < occ <= 1:
            raise ValueError("Atomic occupancy must lie in (0,1].")
        object.__setattr__(self, "fractional_xyz", xyz)
        object.__setattr__(self, "occupancy", occ)

    def to_dict(self):
        return {"element": self.element, "fractional_xyz": list(self.fractional_xyz), "occupancy": self.occupancy}


def direct_cell_matrix(cell_A):
    """Cell vectors as columns; lengths Å, input alpha/beta/gamma degrees."""
    cell = np.asarray(cell_A, dtype=np.float64)
    if cell.shape != (6,) or not np.all(np.isfinite(cell)):
        raise ValueError("cell_A is (a,b,c,alpha_deg,beta_deg,gamma_deg).")
    a, b, c, alpha, beta, gamma = cell
    if min(a, b, c) <= 0 or not np.all((cell[3:] > 0) & (cell[3:] < 180)):
        raise ValueError("Cell lengths/angles are outside their physical range.")
    ca, cb, cg = np.cos(np.deg2rad([alpha, beta, gamma]))
    sg = np.sin(np.deg2rad(gamma))
    cz2 = 1 - cb*cb - ((ca - cb*cg)/sg)**2
    if not math.isfinite(cz2) or cz2 <= 1e-14:
        raise ValueError("Cell metric is singular or non-positive definite.")
    return frozen_array([[a, b*cg, c*cb], [0., b*sg, c*(ca-cb*cg)/sg], [0., 0., c*np.sqrt(cz2)]])


@dataclass(frozen=True)
class PhaseCrystal:
    phase_id: str
    cell_A: tuple[float, float, float, float, float, float]
    sites: tuple[AtomicSite, ...]
    scattering_model: str = "neutral_IT92_no_dispersion"
    source: str = "user_defined_explicit_basis"

    def __post_init__(self):
        if not isinstance(self.phase_id, str) or not self.phase_id:
            raise ValueError("A nonempty phase_id is required.")
        cell = tuple(float(v) for v in self.cell_A)
        direct_cell_matrix(cell)
        sites = tuple(self.sites)
        if not sites or any(not isinstance(site, AtomicSite) for site in sites):
            raise ValueError("Explicit AtomicSite basis is required.")
        if self.scattering_model != "neutral_IT92_no_dispersion":
            raise ValueError("Unsupported scattering model; anomalous/thermal terms are not implemented.")
        groups = {}
        for site in sites:
            groups[site.fractional_xyz] = groups.get(site.fractional_xyz, 0.) + site.occupancy
        if any(total > 1 + 1e-12 for total in groups.values()):
            raise ValueError("Occupancies sharing an atomic site exceed one.")
        object.__setattr__(self, "cell_A", cell)
        object.__setattr__(self, "sites", sites)

    @property
    def direct_basis_A(self):
        return direct_cell_matrix(self.cell_A)

    @property
    def reciprocal_basis_A_inv(self):
        return frozen_array(np.linalg.inv(self.direct_basis_A).T)

    @property
    def cell_volume_A3(self):
        return float(np.linalg.det(self.direct_basis_A))

    @property
    def density_g_cm3(self):
        mass_u = sum(site.occupancy * ATOMIC_MASSES_U[site.element] for site in self.sites)
        return mass_u * ATOMIC_MASS_UNIT_G / (self.cell_volume_A3 * 1e-24)

    @property
    def elemental_mass_fractions(self):
        masses = {}
        for site in self.sites:
            masses[site.element] = masses.get(site.element, 0.) + site.occupancy*ATOMIC_MASSES_U[site.element]
        total = sum(masses.values())
        return {element: mass/total for element, mass in sorted(masses.items())}

    @property
    def fingerprint(self):
        return fingerprint({"phase": self.to_dict(), "atomic_data": ATOMIC_DATA_FINGERPRINT,
                            "atomic_masses_u": ATOMIC_MASSES_U, "u_g": ATOMIC_MASS_UNIT_G})

    def reciprocal_vectors(self, hkls):
        values = np.asarray(hkls)
        if values.ndim < 1 or values.shape[-1] != 3 or not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
            raise ValueError("Miller indices require integer final axis of length 3.")
        return values @ self.reciprocal_basis_A_inv.T

    def structure_factors(self, hkls):
        hkl = np.asarray(hkls)
        gnorm = np.linalg.norm(self.reciprocal_vectors(hkl), axis=-1)
        result = np.zeros(hkl.shape[:-1], dtype=np.complex128)
        absolute_sum = np.zeros_like(gnorm)
        for site in self.sites:
            atom = site.occupancy * neutral_form_factor(site.element, gnorm)
            result += atom * np.exp(2j*np.pi*(hkl @ np.asarray(site.fractional_xyz)))
            absolute_sum += np.abs(atom)
        # Numerical cancellation at exact basis extinctions is not physical signal.
        result = np.where(np.abs(result) <= 64*np.finfo(float).eps*absolute_sum, 0j, result)
        return result

    def to_dict(self):
        return {"phase_id": self.phase_id, "cell_A": list(self.cell_A),
                "sites": [site.to_dict() for site in self.sites],
                "scattering_model": self.scattering_model, "source": self.source}

    @classmethod
    def from_dict(cls, value):
        allowed = {"phase_id", "cell_A", "sites", "scattering_model", "source"}
        if set(value) - allowed:
            raise ValueError("Unknown phase-crystal fields.")
        fields = dict(value)
        fields["sites"] = tuple(AtomicSite(**site) for site in fields["sites"])
        return cls(**fields)


@dataclass(frozen=True)
class ReflectionTable:
    hkls: np.ndarray
    G_crystal_A_inv: np.ndarray
    structure_factor_squared: np.ndarray
    cell_volume_A3: float
    phase_fingerprint: str

    def __post_init__(self):
        hkls = np.asarray(self.hkls)
        vectors = np.asarray(self.G_crystal_A_inv)
        factors = np.asarray(self.structure_factor_squared)
        if hkls.ndim != 2 or hkls.shape[1] != 3 or vectors.shape != hkls.shape or factors.shape != (len(hkls),):
            raise ValueError("Reflection table requires [N,3] indices/vectors and [N] intensities.")
        if not np.all(np.isfinite(hkls)) or not np.all(hkls == np.rint(hkls)) or not np.all(np.isfinite(vectors)):
            raise ValueError("Reflection indices must be integers and reciprocal vectors finite.")
        if not np.all(np.isfinite(factors)) or np.any(factors < 0) or not math.isfinite(self.cell_volume_A3) or self.cell_volume_A3 <= 0:
            raise ValueError("Reflection intensities and cell volume are outside their physical range.")
        object.__setattr__(self, "hkls", frozen_array(self.hkls, dtype=np.int64))
        object.__setattr__(self, "G_crystal_A_inv", frozen_array(self.G_crystal_A_inv))
        object.__setattr__(self, "structure_factor_squared", frozen_array(self.structure_factor_squared))

    @property
    def norms_A_inv(self):
        return np.linalg.norm(self.G_crystal_A_inv, axis=1)


def enumerate_reflections(phase: PhaseCrystal, g_max_A_inv: float, *, candidate_margin_A_inv=0., max_candidates=200000):
    """Enumerate both Friedel signs in a physical reciprocal-radius bound.

    h_i = a_i dot G implies |h_i| <= |a_i|*g_max even for skew cells.
    The configurable count bound rejects excessive work before allocation.
    """
    limit = float(g_max_A_inv) + float(candidate_margin_A_inv)
    if not np.all(np.isfinite([g_max_A_inv, candidate_margin_A_inv])) or g_max_A_inv <= 0 or candidate_margin_A_inv < 0 or limit >= 4:
        raise ValueError("A positive reciprocal bound plus nonnegative margin below 4 Å^-1 is required.")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer.")
    # Direct lengths are already |a_i|. Check before the integer conversion so
    # an unreasonable user cell cannot overflow into an unbounded allocation.
    scaled_lengths = np.asarray(phase.cell_A[:3])*limit
    if np.any(scaled_lengths > (max_candidates-1)/2):
        raise ValueError("Reciprocal enumeration exceeds max_candidates.")
    maxima = [math.ceil(float(value)) for value in scaled_lengths]
    candidate_count = math.prod(int(2*n+1) for n in maxima)
    if candidate_count > max_candidates:
        raise ValueError("Reciprocal enumeration exceeds max_candidates.")
    hkls = np.array(list(itertools.product(*(range(-int(n), int(n)+1) for n in maxima))), dtype=np.int64)
    vectors = phase.reciprocal_vectors(hkls)
    norms = np.linalg.norm(vectors, axis=1)
    selected = (norms > 0) & (norms < 4) & (norms <= limit*(1+8*np.finfo(float).eps))
    hkls, vectors, norms = hkls[selected], vectors[selected], norms[selected]
    factors = np.abs(phase.structure_factors(hkls))**2
    nonzero = factors > 0
    hkls, vectors, norms, factors = hkls[nonzero], vectors[nonzero], norms[nonzero], factors[nonzero]
    order = np.lexsort((hkls[:,2], hkls[:,1], hkls[:,0], norms))
    return ReflectionTable(hkls[order], vectors[order], factors[order], phase.cell_volume_A3, phase.fingerprint)


def built_in_phase(name: str):
    """Nominal demonstration cells, not certified lattice-parameter standards."""
    names = {"cu_fcc": ("Cu", 3.61, "fcc"), "fe_bcc": ("Fe", 2.866, "bcc"),
             "ti_hcp": ("Ti", 2.95, "hcp"), "nb_bcc": ("Nb", 3.3, "bcc")}
    key = name.lower()
    if key not in names:
        raise ValueError(f"Unknown nominal built-in phase {name!r}.")
    element, a, structure = names[key]
    if structure == "fcc":
        positions = ((0,0,0), (0,.5,.5), (.5,0,.5), (.5,.5,0))
        cell = (a,a,a,90.,90.,90.)
    elif structure == "bcc":
        positions = ((0,0,0), (.5,.5,.5))
        cell = (a,a,a,90.,90.,90.)
    else:
        positions = ((0,0,0), (2/3,1/3,.5))
        cell = (a,a,4.683,90.,90.,120.)
    return PhaseCrystal(key, cell, tuple(AtomicSite(element, point) for point in positions),
                        source="nominal_demo_cell_and_explicit_monatomic_basis_v1")
