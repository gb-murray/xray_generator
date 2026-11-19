"""Crystallographic signals through acquisition, cache and native readout."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest

from xray_generator.api import Exposure,SimulationRequest,expected,observe,validate
from xray_generator.experiment.cache import ExpectationCache
from xray_generator.material import ProceduralMaterial
from xray_generator.measurement.integrate import IntegrationOptions
from xray_generator.model import MaterialRecipe,PhaseDefinition
from xray_generator.physics.crystallography import built_in_phase
from xray_generator.physics.detector import PlanarDetector
from xray_generator.physics.texture import TextureConfig
from xray_generator.solvers.crystallographic import CrystallographicRateBackend,PhaseScattering


def backend_and_material(**overrides):
    crystals=[built_in_phase(name) for name in ["cu_fcc","ti_hcp"]]
    specs=[PhaseScattering(crystal,TextureConfig(num_orientations=32,seed=12+i),coherent_domain_nm=3.)
           for i,crystal in enumerate(crystals)]
    options=dict(detector=PlanarDetector((32,32),.1,10.),illuminated_volume_um3=1.,
                 incident_flux_density_photons_s_um2=1e4,pixel_subdivision=4,comparison_subdivision=2)
    options.update(overrides)
    backend=CrystallographicRateBackend(specs,**options)
    phases=[PhaseDefinition(p.phase_id,tuple(p.elemental_mass_fractions),tuple(p.elemental_mass_fractions.values()),p.density_g_cm3)
            for p in crystals]
    recipe=MaterialRecipe.sharp_interface(bounds_mm=((-2.,2.),(-.1,.1),(0.,.001)),phases=phases,
                                         left_fraction=.5,right_fraction=.5)
    return backend,ProceduralMaterial(recipe)


def test_atomic_volume_mixing_and_prepared_immutability():
    backend,material=backend_and_material()
    rates=backend.basis_rates()
    assert rates.shape==(3,32,32) and rates[:2].sum()>0
    state=material.state_at([[0.,0.,.0005]])
    weights=backend.integration_weights(state)[0]
    rho=[p.density_g_cm3 for p in material.recipe.phases]
    target=np.array([.5/r for r in rho]); target/=target.sum()
    np.testing.assert_allclose(weights[:2],target,rtol=1e-12)
    assert not np.allclose(weights[:2],[.5,.5])
    with pytest.raises(AttributeError): backend.wavelength_A=1.
    with pytest.raises(ValueError): rates.setflags(write=True)


def test_physical_backend_connects_acquisition_and_cache():
    backend,material=backend_and_material(background_rate_per_pixel=.2)
    request=SimulationRequest(material,backend,Exposure("physical",0.,.01,0.,0.,.2),
                              IntegrationOptions(atol=.05,rtol=.03))
    cache=ExpectationCache(2*1024**2)
    result=expected(request,cache=cache)
    state=material.state_at([[0.,0.,.0005]])
    target=np.tensordot(backend.integration_weights(state)[0],backend.basis_rates(),axes=(0,0))*.01
    np.testing.assert_allclose(result.total_mean,target,rtol=1e-12,atol=1e-12)
    assert result.profile=="static_kinematic_gaussian_v1"
    assert not validate(request)["calibrated_physics"]
    again=expected(request,cache=cache)
    assert again.cache_hit
    observation,_=observe(request,root_seed=1,cache=cache)
    assert observation.counts.shape==(32,32)
    # Sample is missed, but the detector background remains.
    outside=expected(replace(request,exposure=replace(request.exposure,center_at_zero_mm=4.)),cache=cache)
    np.testing.assert_allclose(outside.total_mean,.002,atol=1e-12)


def test_material_basis_conflicts_and_unresolved_peaks_fail():
    backend,material=backend_and_material()
    value=material.recipe.to_dict()
    value["phases"][0]["density_g_cm3"]*=2
    wrong=ProceduralMaterial(MaterialRecipe.from_dict(value))
    with pytest.raises(ValueError,match="density"):
        validate(SimulationRequest(wrong,backend,Exposure("wrong",0.,1.)))
    specs=[PhaseScattering(p.crystal,p.texture,coherent_domain_nm=33500.) for p in backend.phase_specs]
    with pytest.raises(ValueError,match="undersampled"):
        CrystallographicRateBackend(specs,detector=backend.detector)


def test_equal_volume_flux_scaling_has_no_image_normalization():
    first,_=backend_and_material()
    second,_=backend_and_material(illuminated_volume_um3=2.)
    third,_=backend_and_material(incident_flux_density_photons_s_um2=2e4)
    np.testing.assert_allclose(second.basis_rates(),2*first.basis_rates(),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(third.basis_rates(),2*first.basis_rates(),rtol=1e-12,atol=1e-12)


def test_crystallographic_cli_bundle_retains_truth_and_restarts(tmp_path,capsys):
    from xray_generator.cli import main
    from xray_generator.experiment.bundles import inspect_bundle
    from xray_generator.experiment.provenance import filesystem_path
    backend,material=backend_and_material()
    config=dict(schema_version='xray_generator.crystallographic_experiment/1',profile=backend.profile,
                material=material.recipe.to_dict(),crystallography=backend.to_dict(),
                exposures=[dict(exposure_id='tiny-physical',start_s=0.,stop_s=.01)],
                integration=dict(atol=.05,rtol=.03),readout=dict(root_seed=17,replicate_ids=[701,702]))
    path=tmp_path/'physical.json'; path.write_text(json.dumps(config),encoding='utf-8')
    output=tmp_path/'bundles'; cache=tmp_path/'cache'
    args=['observe',str(path),'--output',str(output),'--cache-dir',str(cache)]
    assert main(args)==0
    first=json.loads(capsys.readouterr().out)
    assert [x['cache_hit'] for x in first['outputs']]==[False,True]
    bundle=inspect_bundle(first['outputs'][0]['path'])
    assert bundle['profile']==backend.profile and bundle['training_truth_available']
    records=list((output/'.runs').glob('*.json')); assert len(records)==1
    saved=json.loads(filesystem_path(records[0]).read_text(encoding='utf-8'))
    assert saved['configuration']['crystallography']['phase_specs'][0]['crystal']['sites']
    assert main(args)==0
    restarted=json.loads(capsys.readouterr().out)
    assert all(item['cache_hit'] and item['reused'] for item in restarted['outputs'])
    assert 'crystallographic_preparation' not in restarted


def test_high_resolution_demo_is_explicit_and_legacy_profile_stays_separate():
    from xray_generator.cli import load_requests
    root=Path(__file__).resolve().parents[1]
    config,requests=load_requests(root/'configs/v1/crystallographic_texture_demo.json')
    backend=requests[0].backend
    assert backend.native_shape==(512,512)
    assert all(p.texture.kind!='random' and p.texture.num_orientations==4096 for p in backend.phase_specs)
    assert config['readout']['replicate_ids']==[701,702]
    assert len(requests)==2 and not validate(requests[0])['calibrated_physics']
    _,legacy=load_requests(root/'configs/v1/reference_scan_fixture.json')
    assert legacy[0].backend.profile=='synthetic_reference_fixture_v1'
