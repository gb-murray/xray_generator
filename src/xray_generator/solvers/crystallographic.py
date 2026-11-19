"""Static kinematic diffraction from persistent representative orientations.

This is an explicitly restricted phase-pattern field, not a positioned-grain
scan. Per-phase responses are computed at equal representative illuminated
volume, then mixed using physical phase volume fractions. A fixed geometry
allows exact positive linear factorization through footprint/time integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..experiment.provenance import file_sha256, fingerprint
from ..physics.crystallography import PhaseCrystal, built_in_phase, enumerate_reflections
from ..physics.detector import PlanarDetector, render_reciprocal_peaks
from ..physics.reciprocal_kernel import coherent_sigma_G_A_inv
from ..physics.texture import TextureConfig, orientation_ensemble
from .contracts import readonly_array

CLASSICAL_ELECTRON_RADIUS_A=2.8179403262e-5


@lru_cache(maxsize=1)
def _orientation_implementation():
    from src.classes_and_functions import orientation,texture
    return {"orientation.py":file_sha256(Path(orientation.__file__)),
            "texture.py":file_sha256(Path(texture.__file__))}


@dataclass(frozen=True)
class PhaseScattering:
    crystal: PhaseCrystal
    texture: TextureConfig
    coherent_domain_nm: float = 10.
    shape_factor: float = .9

    def __post_init__(self):
        if not isinstance(self.crystal,PhaseCrystal) or not isinstance(self.texture,TextureConfig):
            raise TypeError("Explicit crystal and texture definitions are required.")
        coherent_sigma_G_A_inv(self.coherent_domain_nm,self.shape_factor)

    def to_dict(self):
        return {"crystal":self.crystal.to_dict(),"texture":self.texture.to_dict(),
                "coherent_domain_nm":self.coherent_domain_nm,"shape_factor":self.shape_factor}

    @classmethod
    def from_dict(cls,value):
        allowed={"crystal","phase","texture","coherent_domain_nm","shape_factor"}
        if set(value)-allowed or ("crystal" in value)==("phase" in value):
            raise ValueError("Supply one explicit crystal or built-in phase and supported scattering fields.")
        crystal=PhaseCrystal.from_dict(value["crystal"]) if "crystal" in value else built_in_phase(value["phase"])
        return cls(crystal,TextureConfig.from_dict(value.get("texture",{})),
                   value.get("coherent_domain_nm",10.),value.get("shape_factor",.9))


class CrystallographicRateBackend:
    """Corrected finite-domain diffraction; absolute counts remain uncalibrated."""

    profile="static_kinematic_gaussian_v1"
    modality="xrd"
    numerical_error_kind="pixel_refinement_and_nested_quadrature_estimate"
    scale_origin="physical_model_unvalidated_absolute_scale"

    def __setattr__(self,name,value):
        if getattr(self,"_sealed",False):
            raise AttributeError("Crystallographic physical settings are immutable; construct a new backend.")
        object.__setattr__(self,name,value)

    def __init__(self, phase_specs, *, detector=None, wavelength_A=.5166,
                 incident_flux_density_photons_s_um2=1e6, illuminated_volume_um3=1.,
                 polarization="unpolarized", pixel_subdivision=4, comparison_subdivision=2,
                 support_sigma=7.5, max_working_bytes=256*1024**2,
                 background_rate_per_pixel=0.):
        self.phase_specs=tuple(phase_specs)
        if len(self.phase_specs)!=2 or any(not isinstance(p,PhaseScattering) for p in self.phase_specs):
            raise ValueError("This material field currently supports exactly two explicit crystalline phases.")
        if len(set(self.component_ids))!=2: raise ValueError("Phase IDs must be distinct.")
        self.detector=detector or PlanarDetector()
        if not isinstance(self.detector,PlanarDetector): raise TypeError("An explicit PlanarDetector is required.")
        if np.prod(self.detector.shape)>262144:
            raise ValueError("The current native bundle/renderer supports at most 512x512 channels.")
        scalar=[wavelength_A,incident_flux_density_photons_s_um2,illuminated_volume_um3,support_sigma,background_rate_per_pixel]
        if not np.all(np.isfinite(scalar)) or min(scalar[:4])<=0 or background_rate_per_pixel<0:
            raise ValueError("Finite positive geometry/source/volume/support and nonnegative background are required.")
        if polarization not in {"unpolarized","linear_y","linear_z",None}:
            raise ValueError("Unsupported polarization.")
        if any(isinstance(n,bool) or not isinstance(n,int) for n in [pixel_subdivision,comparison_subdivision,max_working_bytes]):
            raise ValueError("Quadrature counts and memory cap must be integers.")
        if not 1<=comparison_subdivision<pixel_subdivision<=32 or max_working_bytes<1048576:
            raise ValueError("Require 1<=comparison nodes<fine nodes<=32 and a positive working budget.")
        self.wavelength_A=float(wavelength_A)
        self.incident_flux_density_photons_s_um2=float(incident_flux_density_photons_s_um2)
        self.illuminated_volume_um3=float(illuminated_volume_um3)
        self.polarization=polarization
        self.pixel_subdivision=pixel_subdivision
        self.comparison_subdivision=comparison_subdivision
        self.support_sigma=float(support_sigma)
        self.max_working_bytes=max_working_bytes
        self.background_rate_per_pixel=float(background_rate_per_pixel)
        self._prepared=None
        maximum_step=self.detector.pixel_pitch_mm/(self.detector.distance_mm*self.wavelength_A*comparison_subdivision)
        for phase in self.phase_specs:
            if coherent_sigma_G_A_inv(phase.coherent_domain_nm,phase.shape_factor)<maximum_step:
                raise ValueError("Coherent peak is undersampled: reduce domain size or increase detector/pixel quadrature resolution.")
        if any(not np.allclose(p.texture.sample_to_lab_wxyz,self.phase_specs[0].texture.sample_to_lab_wxyz,
                               atol=1e-12,rtol=0) for p in self.phase_specs[1:]):
            raise ValueError("All phases in a single sample require the same sample-to-laboratory pose.")
        self._sealed=True

    @property
    def component_ids(self): return tuple(p.crystal.phase_id for p in self.phase_specs)

    @property
    def native_shape(self): return self.detector.shape

    def to_dict(self):
        return {"phase_specs":[p.to_dict() for p in self.phase_specs],"detector":self.detector.to_dict(),
                "wavelength_A":self.wavelength_A,"incident_flux_density_photons_s_um2":self.incident_flux_density_photons_s_um2,
                "illuminated_volume_um3":self.illuminated_volume_um3,"polarization":self.polarization,
                "pixel_subdivision":self.pixel_subdivision,"comparison_subdivision":self.comparison_subdivision,
                "support_sigma":self.support_sigma,"max_working_bytes":self.max_working_bytes,
                "background_rate_per_pixel":self.background_rate_per_pixel}

    @classmethod
    def from_dict(cls,value):
        fields=dict(value)
        phases=tuple(PhaseScattering.from_dict(p) for p in fields.pop("phase_specs"))
        detector=PlanarDetector(**fields.pop("detector",{}))
        return cls(phases,detector=detector,**fields)

    @property
    def fingerprint(self):
        return fingerprint({"backend":"static_normalized_reciprocal_gaussian/1","configuration":self.to_dict(),
                            "phase_data":[p.crystal.fingerprint for p in self.phase_specs],
                            "orientation_implementation":_orientation_implementation(),
                            "classical_electron_radius_A":CLASSICAL_ELECTRON_RADIUS_A})

    def validate_material(self,material):
        if tuple(material.phase_ids)!=self.component_ids:
            raise ValueError("Material/crystallographic phase order differs.")
        for declared,spec in zip(material.recipe.phases,self.phase_specs):
            composition=dict(zip(declared.elements,declared.elemental_mass_fractions))
            if set(composition)!=set(spec.crystal.elemental_mass_fractions) or any(
                not np.isclose(composition[k],v,atol=1e-12,rtol=0) for k,v in spec.crystal.elemental_mass_fractions.items()):
                raise ValueError("Material composition conflicts with the explicit atomic basis.")
            if not np.isclose(declared.density_g_cm3,spec.crystal.density_g_cm3,rtol=1e-10,atol=0):
                raise ValueError("Material density must match the site/cell-derived phase density.")

    def integration_weights(self,state):
        if tuple(state.phase_ids)!=self.component_ids: raise ValueError("Material phase order differs.")
        volume=np.asarray(state.phase_volume_fractions)*state.occupied[:,None]
        # Detector background persists even when the beam misses the sample.
        return np.column_stack((volume,np.ones(len(volume))))

    def _prepare(self):
        if self._prepared is not None: return self._prepared
        images=[]; errors=[]; reports=[]
        # Bound |G(ray)| on pixel AREAS, not only detector-center pixels.
        corners=np.array([[-.5,-.5],[-.5,self.native_shape[1]-.5],
                          [self.native_shape[0]-.5,-.5],[self.native_shape[0]-.5,self.native_shape[1]-.5]])
        g,_,_=self.detector.sample_rays(corners[:,0],corners[:,1],wavelength_A=self.wavelength_A)
        gmax=float(np.linalg.norm(g,axis=1).max())
        for spec in self.phase_specs:
            sigma=float(coherent_sigma_G_A_inv(spec.coherent_domain_nm,spec.shape_factor))
            table=enumerate_reflections(spec.crystal,gmax,candidate_margin_A_inv=self.support_sigma*sigma)
            count=len(table.hkls)*spec.texture.num_orientations
            if count*128+int(np.prod(self.native_shape))*64>self.max_working_bytes:
                raise MemoryError("Reflection/orientation population exceeds the declared working budget.")
            population=orientation_ensemble(spec.texture)
            centers=np.einsum("nij,hj->nhi",population.lab_rotation_matrices,table.G_crystal_A_inv).reshape(-1,3)
            # Flux/um² -> flux/Å² and volume um³ -> Å³. No per-image normalization.
            coefficient=(self.incident_flux_density_photons_s_um2*1e-8
                         *CLASSICAL_ELECTRON_RADIUS_A**2*self.illuminated_volume_um3*1e12
                         /spec.crystal.cell_volume_A3**2)
            strengths=(coefficient*population.volume_weights[:,None]*table.structure_factor_squared[None,:]).reshape(-1)
            common=dict(wavelength_A=self.wavelength_A,detector=self.detector,support_sigma=self.support_sigma,
                        polarization=self.polarization,max_working_bytes=self.max_working_bytes)
            coarse=render_reciprocal_peaks(centers,strengths,sigma,pixel_subdivision=self.comparison_subdivision,**common)
            fine=render_reciprocal_peaks(centers,strengths,sigma,pixel_subdivision=self.pixel_subdivision,**common)
            images.append(fine.image)
            # Per-pixel convergence estimate plus a complete-sphere tail bound
            # conditional on this finite reflection set. It does not bound
            # excluded higher-order reflections or physical-model uncertainty.
            error=2*np.abs(fine.image-coarse.image)+fine.omitted_support_rate_bound
            errors.append(error)
            reports.append({"phase_id":spec.crystal.phase_id,"phase_fingerprint":spec.crystal.fingerprint,
                "orientation_population_fingerprint":population.fingerprint,"orientation_count":spec.texture.num_orientations,
                "reflection_count":len(table.hkls),"reciprocal_centers":count,"candidate_count":fine.candidate_count,
                "rendered_count":fine.rendered_count,"sigma_G_A_inv":sigma,"g_max_A_inv":gmax,
                "rate_sum":float(fine.image.sum()),"coarse_fine_l1_rate_difference":float(np.abs(fine.image-coarse.image).sum()),
                "omitted_support_rate_bound":fine.omitted_support_rate_bound,
                "tail_bound_scope":"enumerated_reflections_only",
                "reflection_cutoff_G_A_inv":gmax+self.support_sigma*sigma,
                "orientation_population_convergence":"not_assessed"})
        images.append(np.full(self.native_shape,self.background_rate_per_pixel)); errors.append(np.zeros(self.native_shape))
        object.__setattr__(self,"_prepared",(readonly_array(np.stack(images)),readonly_array(np.stack(errors)),tuple(reports)))
        return self._prepared

    def basis_rates(self): return self._prepare()[0]

    def basis_errors(self): return self._prepare()[1]

    @property
    def preparation_report(self): return tuple(dict(row) for row in self._prepare()[2])

    def capabilities(self):
        return {"profile":self.profile,"scale_origin":self.scale_origin,"calibrated_physics":False,
            "spatial_geometry":"1D homogenized phase-pattern field; static representative illuminated volume",
            "texture":"persistent volume-weighted orientation population; not positioned grains",
            "kernel":"unit-integral 3D Gaussian in reciprocal cycles/Angstrom",
            "error_scope":"finite reflection set; pixel/space/time estimates; excludes orientation-population and physical-model error",
            "readout":"conditional native Poisson","unsupported":["positioned-grain overlap covariance",
            "dynamic rotation","absorption","anomalous/thermal factors","physical XRF transport",
            "detector PSF","source bandwidth/divergence","dead time","pileup","material evolution"]}
