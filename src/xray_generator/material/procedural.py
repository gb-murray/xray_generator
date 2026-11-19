"""Coordinate-queryable dense-film material, independent of query resolution."""
from __future__ import annotations

import numpy as np

from ..model.conventions import points_mm
from ..model.sample import MaterialRecipe, MaterialStateBatch


class ProceduralMaterial:
    def __init__(self, recipe: MaterialRecipe):
        if not isinstance(recipe, MaterialRecipe):
            raise TypeError("recipe must be a MaterialRecipe")
        self._recipe = recipe

    @property
    def recipe(self):
        return self._recipe

    @property
    def fingerprint(self):
        return self.recipe.fingerprint

    @property
    def recipe_identity(self):
        return self.fingerprint

    @property
    def phase_ids(self):
        return tuple(p.phase_id for p in self.recipe.phases)

    @property
    def x_breakpoints_mm(self):
        return tuple(sorted({v for r in self.recipe.regions for v in r.x_nodes_mm}))

    def first_phase_fraction(self, x_mm):
        """Evaluate x-field only; return zero outside its geometric x support."""
        x = np.asarray(x_mm, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("x_mm must be finite")
        result = np.zeros_like(x)
        for i, region in enumerate(self.recipe.regions):
            inside = ((x >= region.x_nodes_mm[0]) &
                      ((x < region.x_nodes_mm[-1]) if i < len(self.recipe.regions)-1 else (x <= region.x_nodes_mm[-1])))
            result[inside] = np.interp(x[inside], region.x_nodes_mm, region.first_phase_mass_fractions)
        return result

    def state_at(self, points_sample_mm):
        points = points_mm(points_sample_mm)
        bounds = np.asarray(self.recipe.bounds_mm)
        occupied = np.all((points >= bounds[:, 0]) & (points <= bounds[:, 1]), axis=1)
        first = self.first_phase_fraction(points[:, 0])
        mass = np.column_stack((first, 1-first))
        mass[~occupied] = 0.0
        rho_phase = np.array([p.density_g_cm3 for p in self.recipe.phases])
        specific_volume = mass / rho_phase
        inv_rho = specific_volume.sum(axis=1)
        rho = np.divide(1.0, inv_rho, out=np.zeros_like(inv_rho), where=occupied)
        volume = specific_volume * rho[:, None]
        elements = tuple(sorted({e for p in self.recipe.phases for e in p.elements}))
        composition = np.zeros((2, len(elements)))
        for j, phase in enumerate(self.recipe.phases):
            for element, fraction in zip(phase.elements, phase.elemental_mass_fractions):
                composition[j, elements.index(element)] = fraction
        return MaterialStateBatch(self.phase_ids, elements, occupied, mass, volume, mass @ composition, rho)

    def integrate_phase_mass_fraction_x(self, lower_mm, upper_mm):
        """Exact integral of the binary mass-fraction field along x, in mm.

        This is not physical phase mass: variable density must also be included
        for that quantity. It is useful for linear reference-rate fixtures.
        """
        lo, hi = float(lower_mm), float(upper_mm)
        if not np.isfinite([lo, hi]).all() or hi < lo:
            raise ValueError("Require finite lower_mm <= upper_mm")
        total_first = 0.0
        occupied_length = max(0.0, min(hi, self.recipe.bounds_mm[0][1])-max(lo, self.recipe.bounds_mm[0][0]))
        for region in self.recipe.regions:
            a, b = max(lo, region.x_nodes_mm[0]), min(hi, region.x_nodes_mm[-1])
            if b <= a:
                continue
            nodes = np.array([a] + [v for v in region.x_nodes_mm if a < v < b] + [b])
            values = np.interp(nodes, region.x_nodes_mm, region.first_phase_mass_fractions)
            total_first += float(np.sum(np.diff(nodes)*(values[:-1]+values[1:])/2))
        return np.array([total_first, occupied_length-total_first])

    def mass_g(self):
        """Integrate dense-mixture mass analytically over the piecewise field."""
        rho0, rho1 = [p.density_g_cm3 for p in self.recipe.phases]
        alpha, beta = 1/rho1, 1/rho0 - 1/rho1
        rho_integral = 0.0
        for region in self.recipe.regions:
            for xa, xb, wa, wb in zip(region.x_nodes_mm, region.x_nodes_mm[1:],
                                      region.first_phase_mass_fractions, region.first_phase_mass_fractions[1:]):
                ua, ub = alpha+beta*wa, alpha+beta*wb
                delta = ub-ua
                if delta == 0.0:
                    contribution = (xb-xa)/ua
                else:
                    contribution = (xb-xa)*np.log1p(delta/ua)/delta
                rho_integral += contribution
        bounds = self.recipe.bounds_mm
        yz_mm2 = (bounds[1][1]-bounds[1][0])*(bounds[2][1]-bounds[2][0])
        return float(rho_integral*yz_mm2*1e-3)  # mm^3 to cm^3
