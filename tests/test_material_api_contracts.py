"""Cross-layer material and observation invariants for reference fixtures."""
import json

import numpy as np
import pytest

from xray_generator.api import Exposure, Observation, SimulationRequest, expected, validate
from xray_generator.experiment.bundles import inspect_bundle, write_bundle
from xray_generator.material import ProceduralMaterial
from xray_generator.model import MaterialRecipe, MaterialStateBatch, PhaseDefinition
from xray_generator.solvers.reference import ReferenceRateBackend


def _material():
    phases=(PhaseDefinition("A",("A",),(1.,),1.),PhaseDefinition("B",("B",),(1.,),1.))
    return ProceduralMaterial(MaterialRecipe.sharp_interface(
        bounds_mm=((0.,2.),(3.,4.),(5.,6.)),phases=phases,interface_x_mm=1.))


def _backend():
    return ReferenceRateBackend(np.array([[[2.]],[[5.]]]),("A","B"))


def test_reference_uses_declared_slab_and_preserves_partial_illumination():
    material,backend=_material(),_backend()
    inside=SimulationRequest(material,backend,Exposure("inside",0,2,center_at_zero_mm=.5,beam_width_mm=.2))
    np.testing.assert_allclose(expected(inside).total_mean,[[4.]],rtol=1e-12,atol=1e-12)
    partial=SimulationRequest(material,backend,Exposure("partial",0,1,center_at_zero_mm=0,beam_width_mm=.2))
    np.testing.assert_allclose(expected(partial).total_mean,[[1.]],rtol=1e-12,atol=1e-12)
    outside=SimulationRequest(material,backend,Exposure("outside",0,1,center_at_zero_mm=-.5,beam_width_mm=.2))
    np.testing.assert_array_equal(expected(outside).total_mean,[[0.]])


def test_reference_layout_and_expectations_cannot_be_mutated_after_identity():
    backend=_backend()
    identity=backend.fingerprint
    with pytest.raises(ValueError):
        backend.component_rates.setflags(write=True)
    result=expected(SimulationRequest(_material(),backend,Exposure("stable",0,1,center_at_zero_mm=.5)))
    with pytest.raises(ValueError):
        result.total_mean.setflags(write=True)
    assert backend.fingerprint==identity


def test_reference_rejects_swapped_phase_identity_even_with_matching_shape():
    material,backend=_material(),_backend()
    with pytest.raises(ValueError):
        validate(SimulationRequest(material,ReferenceRateBackend(backend.component_rates,("B","A")),Exposure("bad",0,1)))
    state=material.state_at([[.5,3.5,5.5]])
    swapped=MaterialStateBatch(("B","A"),state.elements,state.occupied,
                               state.phase_mass_fractions,state.phase_volume_fractions,
                               state.elemental_mass_fractions,state.density_g_cm3)
    with pytest.raises(ValueError):
        backend.local_rate(swapped)


def test_fractional_evaluation_budget_rejected_before_integration():
    with pytest.raises(ValueError):
        validate(SimulationRequest(_material(),_backend(),Exposure("budget",0,1),max_total_evaluations=1.5))


def test_empty_complete_bundle_does_not_count_as_resumable_output(tmp_path):
    (tmp_path/"manifest.json").write_text(json.dumps({"status":"complete","checksums":{}}),encoding="utf-8")
    with pytest.raises(ValueError):
        inspect_bundle(tmp_path)


def test_bundle_rejects_observation_expectation_identity_and_layout_mismatch(tmp_path):
    result=expected(SimulationRequest(_material(),_backend(),Exposure("bundle",0,1,center_at_zero_mm=.5)))
    wrong_identity=Observation(np.array([[2]],dtype=np.int64),"other",result.expectation_id,"r0")
    with pytest.raises(ValueError):
        write_bundle(tmp_path,wrong_identity,result,root_seed=17)
    wrong_layout=Observation(np.zeros((2,2),dtype=np.int64),result.exposure_id,result.expectation_id,"r1")
    with pytest.raises(ValueError):
        write_bundle(tmp_path,wrong_layout,result,root_seed=17)
