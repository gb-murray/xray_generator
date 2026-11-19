"""Analytic and dense-reference checks for synthetic acquisition fixtures."""
from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from ..api import Exposure, SimulationRequest, expected
from ..experiment.cache import ExpectationCache
from ..experiment.runtime import runtime_identity
from ..material import ProceduralMaterial
from ..model import MaterialRecipe, PhaseDefinition
from ..solvers.reference import ReferenceRateBackend


def make_boundary_request():
    phases=(PhaseDefinition("A",("fixture-A",),(1.,),1.),PhaseDefinition("B",("fixture-B",),(1.,),1.))
    recipe=MaterialRecipe.sharp_interface(bounds_mm=((-2.,2.),(-.1,.1),(0.,.001)),phases=phases)
    backend=ReferenceRateBackend(np.array([[10.,2.,.1],[2.,20.,.2]]),("A","B"),modality="xrf")
    return SimulationRequest(ProceduralMaterial(recipe),backend,Exposure("boundary",0.,1.,-1.,2.,2.))


def dense_midpoint_reference(request,*,space_nodes=512,time_nodes=128):
    """Independent fixed-grid calculation for the *linear synthetic* backend.

    Push the linear reference mixing through integration; do not use this shortcut
    for nonlinear crystallography or attenuation. No adaptive integrator is called.
    """
    exp=request.exposure
    times=exp.start_s+(np.arange(time_nodes)+.5)*(exp.stop_s-exp.start_s)/time_nodes
    offsets=((np.arange(space_nodes)+.5)/space_nodes-.5)*exp.beam_width_mm
    fractions=np.zeros(len(request.backend.component_ids))
    bounds=np.asarray(request.material.recipe.bounds_mm)
    y,z=bounds[1:].mean(axis=1)
    for t in times:
        xs=exp.center_at_zero_mm+exp.velocity_mm_s*t+offsets
        state=request.material.state_at(np.column_stack((xs,np.full_like(xs,y),np.full_like(xs,z))))
        fractions+=(np.asarray(state.phase_mass_fractions)*state.occupied[:,None]).mean(axis=0)/time_nodes
    values=fractions @ request.backend.component_rates.reshape(len(fractions),-1)
    return values.reshape(request.backend.native_shape)*(exp.stop_s-exp.start_s)*exp.shutter_fraction


def poisson_kl(reference,candidate):
    ref=np.asarray(reference,dtype=float); cand=np.asarray(candidate,dtype=float)
    if ref.shape!=cand.shape or not np.all(np.isfinite([ref,cand])) or np.any(ref<0) or np.any(cand<0):
        raise ValueError("Matching finite nonnegative native means are required.")
    if np.any((ref>0)&(cand==0)): return float("inf")
    positive=ref>0
    terms=cand-ref
    terms[positive]+=ref[positive]*np.log(ref[positive]/cand[positive])
    return max(0.,float(terms.sum()))


def run_development_gates():
    request=make_boundary_request()
    start=time.perf_counter()
    fast=expected(request)
    elapsed=time.perf_counter()-start
    analytic=request.backend.component_rates.mean(axis=0)
    grids=[dense_midpoint_reference(request,space_nodes=n,time_nodes=n//4) for n in [128,256,512]]
    cache=ExpectationCache(100000)
    cold=expected(request,cache=cache)
    warm=expected(replace(request,exposure=replace(request.exposure,exposure_id="same-physics")),cache=cache)
    absolute_error=float(np.max(np.abs(fast.total_mean-analytic)))
    grid_error=max(float(np.max(np.abs(v-analytic))) for v in grids)
    kl=poisson_kl(grids[-1],fast.total_mean)
    return {"schema_version":"xray_generator.development_gate_report/1",
            "scope":"analytic synthetic-reference boundary fixture only",
            "profile":fast.profile,"runtime_identity":runtime_identity(),
            "material_recipe_fingerprint":request.material.fingerprint,
            "reference_backend_fingerprint":request.backend.fingerprint,
            "expectation_id":fast.expectation_id,"numerical_error_kind":fast.numerical_error_kind,
            "physical_profile_certified":False,"posterior_calibration_run":False,
            "gates":[
                {"id":"G10","status":"pass" if absolute_error<1e-8 else "fail","metric":"analytic moving-boundary max absolute event error","value":absolute_error,"limit":1e-8},
                {"id":"G19","status":"pass" if grid_error<1e-8 else "fail","metric":"three fixed-grid refinements versus analytic boundary","value":grid_error,"limit":1e-8},
                {"id":"G20","status":"pass" if warm.cache_hit and np.array_equal(cold.total_mean,warm.total_mean) else "fail","metric":"cold/warm expectation identity; distinct exposure IDs"},
                {"id":"G21","status":"fail" if kl>=.01 else "not_run","reason":"This synthetic fixture does not assess physical-model or inference accuracy.",
                 "deterministic_fixture":{"status":"pass" if kl<.01 else "fail","metric":"conditional native Poisson KL","value":kl,"limit":.01}},
                {"id":"G16","status":"not_run","reason":"Experimental validation is outside this synthetic fixture."}],
            "resources":{"adaptive_seconds":elapsed,"adaptive_function_evaluations":fast.evaluations,
                         "dense_reference_resolutions":[[128,32],[256,64],[512,128]],
                         "compact_recipe_json_bytes":len(str(request.material.recipe.to_dict()).encode()),
                         "cache_memory_bytes":cache.bytes_used,"full_process_memory_cap_certified":False}}
