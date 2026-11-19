"""Independent amount, query-order and bounded lazy-population fixtures."""
import copy
import json
import math

import numpy as np
import pytest

from xray_generator.model import PhaseDefinition, FractionRegion, MaterialRecipe
from xray_generator.model.conventions import q_A_inv_from_G_nm_inv, wavelength_A_from_energy_keV
from xray_generator.material import ProceduralMaterial, LazyGrainGenerator


def phases():
    return (PhaseDefinition("A", ("A",), (1.,), 1.),
            PhaseDefinition("B", ("B",), (1.,), 2.))


def linear_material():
    return ProceduralMaterial(MaterialRecipe(((0.,1.),(0.,1.),(0.,1.)), phases(),
                                            (FractionRegion((0.,1.),(0.,1.)),)))


def pure_material():
    return ProceduralMaterial(MaterialRecipe.sharp_interface(
        bounds_mm=((-0.7,0.9),(-0.2,0.3),(0.,0.01)), phases=phases(), interface_x_mm=0.13))


def test_mass_volume_element_closure_and_outside_support():
    material = linear_material()
    state = material.state_at([[0.25,0.5,0.5], [1.,1.,1.], [-0.1,0.5,0.5]])
    np.testing.assert_array_equal(state.occupied,[True,True,False])
    np.testing.assert_allclose(state.phase_mass_fractions,[[.25,.75],[1,0],[0,0]])
    np.testing.assert_allclose(state.phase_volume_fractions,[[.4,.6],[1,0],[0,0]])
    np.testing.assert_allclose(state.elemental_mass_fractions,state.phase_mass_fractions)
    np.testing.assert_allclose(state.density_g_cm3,[1.6,1.,0.])
    # rho(x)=2/(1+x) g/cm^3; this is one cubic millimetre.
    assert material.mass_g() == pytest.approx(2*math.log(2)*1e-3,rel=1e-13)
    np.testing.assert_allclose(material.integrate_phase_mass_fraction_x(-1,2), [.5,.5])


def test_piecewise_field_preserves_discontinuity_and_exact_integral():
    material = pure_material()
    eps = 1e-8
    batch = material.state_at([[.13-eps,0,.005],[.13,0,.005],[.13+eps,0,.005]])
    np.testing.assert_array_equal(batch.phase_mass_fractions,[[1,0],[0,1],[0,1]])
    np.testing.assert_allclose(material.integrate_phase_mass_fraction_x(-.7,.9),[.83,.77])
    assert material.mass_g() == pytest.approx((.83*1+.77*2)*.5*.01*1e-3)
    assert material.x_breakpoints_mm == (-.7,.13,.9)


def test_material_queries_are_order_and_batch_invariant_with_small_recipe():
    material = linear_material()
    rng = np.random.default_rng(9)
    points = rng.uniform(-.2,1.2,size=(1001,3))
    all_states = material.state_at(points)
    order = rng.permutation(len(points))
    shuffled = material.state_at(points[order])
    np.testing.assert_array_equal(shuffled.phase_mass_fractions,all_states.phase_mass_fractions[order])
    pieces = [material.state_at(p).density_g_cm3 for p in np.array_split(points,17)]
    np.testing.assert_array_equal(np.concatenate(pieces),all_states.density_g_cm3)
    assert len(json.dumps(material.recipe.to_dict())) < 1500
    assert material.state_at(np.empty((0,3))).phase_mass_fractions.shape == (0,2)


def test_recipe_roundtrip_fingerprint_and_immutable_records():
    material = linear_material()
    record = material.recipe.to_dict()
    restored = ProceduralMaterial(MaterialRecipe.from_dict(json.loads(json.dumps(record))))
    assert restored.fingerprint == material.fingerprint
    modified = copy.deepcopy(record)
    modified["regions"][0]["first_phase_mass_fractions"][0] = .01
    assert MaterialRecipe.from_dict(modified).fingerprint != material.fingerprint
    state = material.state_at([[.5,.5,.5]])
    with pytest.raises(ValueError):
        state.phase_mass_fractions[0,0] = 0
    with pytest.raises(ValueError):
        state.phase_mass_fractions.setflags(write=True)
    record["bounds_mm"][0][0] = -10
    assert material.recipe.bounds_mm[0][0] == 0


@pytest.mark.parametrize("bad", [
    {"bounds_mm": ((1,0),(0,1),(0,1))},
    {"regions": (FractionRegion((0,.9),(0,1)),)},
    {"regions": (FractionRegion((0,.4),(0,1)),FractionRegion((.5,1),(0,1)))},
])
def test_invalid_material_geometry_rejected(bad):
    fields = dict(bounds_mm=((0,1),(0,1),(0,1)), phases=phases(),regions=(FractionRegion((0,1),(0,1)),))
    fields.update(bad)
    with pytest.raises(ValueError):
        MaterialRecipe(**fields)


def test_invalid_composition_and_unknown_schema_rejected():
    with pytest.raises(ValueError):
        FractionRegion((0,1),(-.1,1))
    with pytest.raises(ValueError):
        PhaseDefinition("bad",("A",),(.5,),1.)
    with pytest.raises(ValueError):
        MaterialRecipe.from_dict({**linear_material().recipe.to_dict(),"dense_state": []})
    with pytest.raises(ValueError):
        linear_material().state_at([[float("nan"),0,0]])


def _records(batches):
    result = {}
    for batch in batches:
        for i, name in enumerate(batch.grain_ids):
            assert name not in result
            result[name] = (batch.phase_ids[i],batch.bounds_mm[i],batch.orientations_wxyz[i])
    return result


def test_lazy_grains_match_complete_reference_and_neighbor_support():
    generator = LazyGrainGenerator(pure_material(),cell_width_mm=.3,seed=371)
    complete = generator.complete_reference()
    assert len(complete.grain_ids) == 7  # six ownership columns; interface splits one
    assert complete.volume_mm3.sum() == pytest.approx(1.6*.5*.01)
    phase_a = complete.volume_mm3[np.array(complete.phase_ids)=="A"].sum()
    assert phase_a == pytest.approx(.83*.5*.01)
    # Narrow window intersects a grain whose center is outside the query.
    support = np.array([[-.69,-.68],[-.1,.1],[.001,.009]])
    full = _records([complete])
    expected = {name:value for name,value in full.items()
                if np.all(np.minimum(value[1][:,1],support[:,1]) > np.maximum(value[1][:,0],support[:,0]))}
    lazy = _records(generator.iter_grains(support,batch_size=1))
    assert lazy.keys() == expected.keys()
    assert any(not support[0,0] <= value[1][0].mean() <= support[0,1] for value in lazy.values())
    for name in lazy:
        assert lazy[name][0] == expected[name][0]
        np.testing.assert_array_equal(lazy[name][1],expected[name][1])
        np.testing.assert_array_equal(lazy[name][2],expected[name][2])
    np.testing.assert_allclose(np.linalg.norm(complete.orientations_wxyz,axis=1),1.,atol=1e-14)


def test_lazy_grain_query_batching_revisit_and_random_supports():
    material = pure_material()
    gen = LazyGrainGenerator(material,.17,seed=5)
    full = _records([gen.complete_reference()])
    np_rng = np.random.default_rng(21)
    for _ in range(25):
        lo,hi = sorted(np_rng.uniform(-1,1.1,2))
        if lo == hi:
            continue
        support = np.array([[lo,hi],[-.4,.8],[-.1,.1]])
        small = _records(gen.iter_grains(support,batch_size=1))
        large = _records(gen.iter_grains(support,batch_size=13))
        expected = {name:value for name,value in full.items()
                    if np.all(np.minimum(value[1][:,1],support[:,1]) > np.maximum(value[1][:,0],support[:,0]))}
        assert small.keys() == large.keys() == expected.keys()
        for name in small:
            np.testing.assert_array_equal(small[name][2],large[name][2])
            np.testing.assert_array_equal(small[name][2],full[name][2])
    # New generator after all transient query data are discarded regenerates identity.
    again = LazyGrainGenerator(material,.17,seed=5).complete_reference()
    np.testing.assert_array_equal(again.orientations_wxyz,gen.complete_reference().orientations_wxyz)
    with pytest.raises(ValueError):
        gen.complete_reference(maximum_grains=1)


def test_lazy_grains_reject_unvalidated_mixed_or_graded_assignment():
    with pytest.raises(ValueError,match="pure constant"):
        LazyGrainGenerator(linear_material(),.1)
    with pytest.raises(ValueError):
        LazyGrainGenerator(pure_material(),0.)


def test_reciprocal_and_energy_units_are_explicit():
    assert q_A_inv_from_G_nm_inv(10) == pytest.approx(2*np.pi)
    assert wavelength_A_from_energy_keV(12.398419843320026) == pytest.approx(1.)
    with pytest.raises(ValueError):
        wavelength_A_from_energy_keV(0)
