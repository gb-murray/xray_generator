"""Forward acquisition for reference fixtures and static crystallographic fields."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .experiment.cache import ExpectationCache
from .experiment.provenance import fingerprint
from .experiment.runtime import runtime_identity
from .measurement.integrate import IntegrationOptions, integrate_vector
from .measurement.readout import poisson_readout
from .material import ProceduralMaterial
from .solvers.contracts import readonly_array
from .solvers.reference import ReferenceRateBackend
from .solvers.crystallographic import CrystallographicRateBackend


class ResourceLimitError(RuntimeError):
    """The requested accuracy could not be attempted within the work budget."""


@dataclass(frozen=True)
class Exposure:
    exposure_id: str
    start_s: float
    stop_s: float
    center_at_zero_mm: float = 0.0
    velocity_mm_s: float = 0.0
    beam_width_mm: float = 0.02
    shutter_fraction: float = 1.0

    def __post_init__(self):
        values = [self.start_s, self.stop_s, self.center_at_zero_mm,
                  self.velocity_mm_s, self.beam_width_mm, self.shutter_fraction]
        if not isinstance(self.exposure_id,str) or not self.exposure_id or not np.all(np.isfinite(values)):
            raise ValueError("Exposure identity and finite parameters are required.")
        if self.stop_s <= self.start_s or self.beam_width_mm <= 0:
            raise ValueError("Positive exposure duration and beam width are required.")
        if not 0 <= self.shutter_fraction <= 1:
            raise ValueError("Shutter fraction is outside [0,1].")

    def physical_parameters(self):
        values = asdict(self)
        values.pop("exposure_id")
        return values


@dataclass(frozen=True)
class SimulationRequest:
    material: object
    backend: ReferenceRateBackend | CrystallographicRateBackend
    exposure: Exposure
    accuracy: IntegrationOptions = IntegrationOptions()
    max_total_evaluations: int = 100000


@dataclass(frozen=True)
class ExpectedMeasurement:
    total_mean: np.ndarray
    numerical_error: np.ndarray
    expectation_id: str
    exposure_id: str
    cache_hit: bool
    evaluations: int
    profile: str = "synthetic_reference_fixture_v1"
    numerical_error_kind: str = "nested_quadrature_estimate"

    def __post_init__(self):
        mean = readonly_array(self.total_mean)
        error = readonly_array(self.numerical_error)
        if mean.shape != error.shape:
            raise ValueError("Expectation/error shapes differ.")
        object.__setattr__(self, "total_mean", mean)
        object.__setattr__(self, "numerical_error", error)


@dataclass(frozen=True)
class Observation:
    counts: np.ndarray
    exposure_id: str
    expectation_id: str
    readout_id: str


def validate(request: SimulationRequest):
    if not isinstance(request.material,ProceduralMaterial):
        raise ValueError("This release requires a supported compact ProceduralMaterial provider.")
    if not isinstance(request.accuracy,IntegrationOptions):
        raise ValueError("A validated IntegrationOptions record is required.")
    if not isinstance(request.backend,(ReferenceRateBackend,CrystallographicRateBackend)):
        raise ValueError("Unsupported native-rate backend.")
    if tuple(request.material.phase_ids) != request.backend.component_ids:
        raise ValueError("Material phase order differs from reference component order.")
    if isinstance(request.max_total_evaluations, bool) or not isinstance(request.max_total_evaluations,(int,np.integer)) or request.max_total_evaluations < 1:
        raise ValueError("A positive evaluation budget is required.")
    if isinstance(request.backend,CrystallographicRateBackend):
        request.backend.validate_material(request.material)
        return request.backend.capabilities()
    return {"profile": request.backend.profile, "scale_origin": "synthetic_reference_fixture",
            "calibrated_physics": False, "spatial_geometry": "1D extruded film, rectangular beam",
            "readout": "conditional native Poisson", "unsupported": ["rotation", "physical XRF transport",
            "calibrated XRD", "dead time", "pileup", "material evolution"]}


def expectation_dependencies(request):
    return {"engine": "positive_linear_basis_acquisition/2", "material": request.material.fingerprint,
            "runtime": runtime_identity(),
            "backend": request.backend.fingerprint, "exposure": request.exposure.physical_parameters(),
            "accuracy": asdict(request.accuracy), "max_total_evaluations": request.max_total_evaluations,
            "dtype": "float64"}


def expected(request: SimulationRequest, *, cache: ExpectationCache | None = None):
    """Integrate the same material recipe; a noise key never enters this function."""
    validate(request)
    dependencies = expectation_dependencies(request)
    key = fingerprint(dependencies)
    exposure = request.exposure
    error_kind=getattr(request.backend,"numerical_error_kind","nested_quadrature_estimate")
    if cache is not None:
        hit = cache.get(key)
        valid_hit=(hit is not None and hit.values.shape==request.backend.native_shape
                   and hit.error.shape==request.backend.native_shape
                   and hit.metadata.get("profile")==request.backend.profile
                   and hit.metadata.get("error_kind")==error_kind
                   and fingerprint(hit.metadata.get("dependencies"))==fingerprint(dependencies))
        if valid_hit and np.all(hit.error <= request.accuracy.atol+request.accuracy.rtol*np.abs(hit.values)):
            return ExpectedMeasurement(hit.values, hit.error, key, exposure.exposure_id, True, 0,
                                       profile=request.backend.profile,numerical_error_kind=error_kind)
    # Fixed-response fields are exactly linear in these positive coefficients.
    # Integrate a few weights, rather than a full detector image at every node.
    rates=request.backend.basis_rates()
    basis_errors=request.backend.basis_errors()
    basis_count=len(rates)
    spatial_error_max=np.zeros(basis_count)
    work = [0]
    options = request.accuracy
    scale=max(float(np.max(rates+basis_errors))*basis_count,1e-300)
    coefficient_atol=options.atol/scale
    outer_options=IntegrationOptions(atol=coefficient_atol*.25,rtol=options.rtol*.25,
                                     max_evaluations=options.max_evaluations,
                                     max_intervals=options.max_intervals,max_depth=options.max_depth,
                                     max_working_bytes=max(1,options.max_working_bytes//2))
    inner_options = IntegrationOptions(atol=coefficient_atol*.25/max(exposure.stop_s-exposure.start_s,1e-12),
                                       rtol=options.rtol*.25,
                                       max_evaluations=options.max_evaluations,
                                       max_intervals=options.max_intervals, max_depth=options.max_depth,
                                       max_working_bytes=max(1,options.max_working_bytes//2))

    def consume():
        work[0] += 1
        if work[0] > request.max_total_evaluations:
            raise ResourceLimitError("Total spatial/time evaluation budget exhausted; no example accepted.")

    def instantaneous(t):
        consume()
        center = exposure.center_at_zero_mm + exposure.velocity_mm_s*t
        left, right = center-exposure.beam_width_mm/2, center+exposure.beam_width_mm/2

        def local(x):
            consume()
            # The fixture is extruded in y/z; query the declared slab interior.
            bounds = np.asarray(request.material.recipe.bounds_mm)
            y, z = bounds[1:].mean(axis=1)
            state = request.material.state_at(np.array([[x,y,z]]))
            return request.backend.integration_weights(state)[0]/exposure.beam_width_mm

        result = integrate_vector(local,left,right,options=inner_options,
                                  breakpoints=[x for x in request.material.x_breakpoints_mm if left<x<right])
        np.maximum(spatial_error_max,result.error,out=spatial_error_max)
        return result.value*exposure.shutter_fraction

    boundaries=[]
    if exposure.velocity_mm_s:
        for x in request.material.x_breakpoints_mm:
            for shift in [-exposure.beam_width_mm/2,exposure.beam_width_mm/2]:
                boundaries.append((x-exposure.center_at_zero_mm-shift)/exposure.velocity_mm_s)
    if exposure.shutter_fraction == 0:
        coefficients=np.zeros(basis_count); coefficient_errors=np.zeros(basis_count)
    else:
        result=integrate_vector(instantaneous,exposure.start_s,exposure.stop_s,
                                options=outer_options,breakpoints=[t for t in boundaries if exposure.start_s<t<exposure.stop_s])
        coefficients=result.value
        coefficient_errors=result.error+spatial_error_max*(exposure.stop_s-exposure.start_s)*exposure.shutter_fraction
    values=np.tensordot(coefficients,rates,axes=(0,0))
    errors=(np.tensordot(coefficient_errors,rates+basis_errors,axes=(0,0))
            +np.tensordot(np.abs(coefficients),basis_errors,axes=(0,0)))
    tolerance=options.atol+options.rtol*np.abs(values)
    if np.any(errors>tolerance):
        raise ResourceLimitError("Combined detector/spatial/time error estimate exceeds the requested accuracy; refine pixel quadrature or integration.")
    measurement=ExpectedMeasurement(values,errors,key,exposure.exposure_id,False,work[0],
                                    profile=request.backend.profile,numerical_error_kind=error_kind)
    if cache is not None:
        cache.put(key,measurement.total_mean,error=measurement.numerical_error,
                  metadata={"dependencies":dependencies,"profile":measurement.profile,
                            "error_kind":measurement.numerical_error_kind})
    return measurement


def observe(request, *, root_seed=17, readout_id="0", cache=None):
    measurement=expected(request,cache=cache)
    counts=poisson_readout(measurement.total_mean,root_seed=root_seed,
                          readout_id=(request.exposure.exposure_id,measurement.expectation_id,str(readout_id)))
    return Observation(counts,request.exposure.exposure_id,measurement.expectation_id,str(readout_id)),measurement
