"""Orientation fixtures preserve physical hkl links and specimen identity."""
from dataclasses import replace

import numpy as np
import pytest

from src.classes_and_functions.orientation import quaternion_to_rotation_matrix
from xray_generator.physics.texture import TextureConfig,orientation_ensemble


def test_seed_repeatability_and_recipe_round_trip():
    config=TextureConfig(num_orientations=200,seed=42)
    first=orientation_ensemble(config)
    second=orientation_ensemble(TextureConfig.from_dict(config.to_dict()))
    assert np.array_equal(first.quaternions_wxyz,second.quaternions_wxyz)
    assert first.fingerprint==second.fingerprint
    assert not np.array_equal(first.quaternions_wxyz,orientation_ensemble(replace(config,seed=43)).quaternions_wxyz)
    assert np.allclose(first.volume_weights,1/200)
    for values in [first.quaternions_wxyz,first.volume_weights,first.rotation_matrices,first.lab_rotation_matrices]:
        with pytest.raises(ValueError):
            values.setflags(write=True)


def test_haar_population_has_no_preferred_axis():
    ensemble=orientation_ensemble(TextureConfig(num_orientations=20000,seed=17))
    directions=ensemble.rotation_matrices[:,:,2]
    assert np.max(np.abs(directions.mean(axis=0))) < .02
    assert np.allclose(directions.T@directions/len(directions),np.eye(3)/3,atol=.015)


def test_fiber_aligns_real_crystal_axis_and_rotates_all_hkls_together():
    ensemble=orientation_ensemble(TextureConfig(kind="fiber",num_orientations=1500,
        crystal_direction=(1,0,0),sample_direction=(0,1,0),spread_deg=3,seed=99))
    rotations=ensemble.rotation_matrices
    aligned=rotations[:,:,0]
    beta=np.arccos(np.clip(aligned[:,1],-1,1))
    assert np.mean(beta) < np.deg2rad(5)
    assert np.quantile(beta,.99)<np.deg2rad(11)
    # Orthogonal crystal reciprocal vectors remain orthogonal for each grain;
    # texture cannot independently paint azimuthal profiles onto their rings.
    first=rotations@np.array([1.,0.,0.])
    second=rotations@np.array([0.,2.,0.])
    assert np.allclose(np.einsum("ni,ni->n",first,second),0,atol=1e-14)
    assert np.allclose(np.linalg.norm(second,axis=1),2,atol=1e-14)


def test_sample_pose_rotates_same_population_exactly_once():
    base=TextureConfig(kind="fiber",num_orientations=100,seed=81)
    pose=(np.sqrt(.5),np.sqrt(.5),0,0)
    first=orientation_ensemble(base)
    rotated=orientation_ensemble(replace(base,sample_to_lab_wxyz=pose))
    assert np.array_equal(first.quaternions_wxyz,rotated.quaternions_wxyz)
    expected=np.einsum("ij,njk->nik",quaternion_to_rotation_matrix(np.asarray(pose)),first.rotation_matrices)
    assert np.allclose(rotated.lab_rotation_matrices,expected,atol=1e-14)
    assert rotated.fingerprint != first.fingerprint


def test_partial_fiber_restricts_spin_without_collapsing_misalignment():
    ensemble=orientation_ensemble(TextureConfig(kind="partial_fiber",num_orientations=1000,seed=5,
        spread_deg=2,spin_width_deg=20,spin_center_deg=0))
    rotations=ensemble.rotation_matrices
    # With z fiber and a narrow spin window, crystal x remains near sample x.
    assert np.mean(rotations[:,0,0])>.97
    assert np.mean(rotations[:,2,2])>.99
    assert np.std(rotations[:,0,2])>.01


@pytest.mark.parametrize("fraction",[0,1e-12,.4,1-1e-12,1])
def test_isotropic_mixture_handles_empty_component_batches(fraction):
    config=TextureConfig(kind="fiber",num_orientations=1,seed=4,isotropic_fraction=fraction)
    ensemble=orientation_ensemble(config)
    assert ensemble.quaternions_wxyz.shape==(1,4)
    assert np.linalg.norm(ensemble.quaternions_wxyz)==pytest.approx(1)


def test_volume_weights_normalize_physical_amount_even_for_large_values():
    config=TextureConfig(num_orientations=3)
    ensemble=orientation_ensemble(config,volume_weights=[0,1e308,1e308])
    assert np.allclose(ensemble.volume_weights,[0,.5,.5])
    with pytest.raises(ValueError):
        orientation_ensemble(config,volume_weights=[0,-1,2])
    with pytest.raises(ValueError):
        orientation_ensemble(config,volume_weights=[0,0,0])


@pytest.mark.parametrize("values",[{"num_orientations":0},{"kind":"cosmetic_ring"},
    {"sample_to_lab_wxyz":(0,0,0,0)},{"seed":-1},{"spread_deg":0}])
def test_invalid_texture_configuration_rejected(values):
    with pytest.raises(ValueError):
        TextureConfig(**values)
