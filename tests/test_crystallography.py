"""Physical fixtures for explicit-cell diffraction, independent of legacy rendering."""
import itertools

import numpy as np
import pytest

from xray_generator.physics.crystallography import (
    ATOMIC_MASS_UNIT_G, AtomicSite, PhaseCrystal, built_in_phase,
    direct_cell_matrix, enumerate_reflections, neutral_form_factor,
)


def test_it92_official_neutral_iron_example_and_q_dependence():
    # Gemmi's official scattering documentation uses Fe and sin(theta)^2/lambda^2=.4.
    # These are the original neutral IT92 coefficients, without f(0)=Z rescaling.
    assert neutral_form_factor("Fe", 2*np.sqrt(.4)) == pytest.approx(9.303603, abs=1e-6)
    for element, z in [("Ti",22),("Fe",26),("Cu",29),("Nb",41)]:
        values = neutral_form_factor(element, np.linspace(0,3.9,100))
        assert values[0] == pytest.approx(z, abs=.05)
        assert np.all(np.diff(values) < 0)
        assert values[-1] > 0


@pytest.mark.parametrize("g", [-1,np.inf,np.nan,4])
def test_atomic_data_domain_is_enforced(g):
    with pytest.raises(ValueError):
        neutral_form_factor("Cu",g)


def test_atomic_data_subset_does_not_silently_invent_other_elements():
    with pytest.raises(ValueError):
        neutral_form_factor("Au",1)


@pytest.mark.parametrize("name,forbidden,allowed,multiplicity", [
    ("cu_fcc",[1,0,0],[1,1,1],4),
    ("fe_bcc",[1,0,0],[1,1,0],2),
    ("nb_bcc",[1,0,0],[1,1,0],2),
    ("ti_hcp",[0,0,1],[0,0,2],2),
])
def test_explicit_atomic_basis_extinctions(name,forbidden,allowed,multiplicity):
    phase = built_in_phase(name)
    assert phase.structure_factors(forbidden) == 0
    g = np.linalg.norm(phase.reciprocal_vectors(allowed))
    amplitude = multiplicity*neutral_form_factor(phase.sites[0].element,g)
    assert abs(phase.structure_factors(allowed)) == pytest.approx(amplitude,rel=1e-14)
    table = enumerate_reflections(phase,1)
    values = set(map(tuple,table.hkls.tolist()))
    assert tuple(forbidden) not in values
    assert tuple(allowed) in values


def test_general_cell_duality_and_analytic_hexagonal_metric():
    phase = built_in_phase("ti_hcp")
    assert np.allclose(phase.direct_basis_A.T @ phase.reciprocal_basis_A_inv,np.eye(3),atol=1e-14)
    hkl = np.asarray([[1,0,0],[1,1,0],[2,-1,3],[-3,2,1]])
    h,k,l = hkl.T
    expected = 4*(h*h+h*k+k*k)/(3*phase.cell_A[0]**2) + l*l/phase.cell_A[2]**2
    assert np.allclose(np.sum(phase.reciprocal_vectors(hkl)**2,axis=1),expected,rtol=1e-14)
    # A fully skew cell is not reducible to independent orthogonal spacings.
    direct = direct_cell_matrix((3.2,4.1,5.7,72,85,104))
    reciprocal = np.linalg.inv(direct).T
    assert np.allclose(direct.T@reciprocal,np.eye(3),atol=1e-14)


@pytest.mark.parametrize("cell", [(1,1,1,0,90,90),(1,1,1,10,10,170),(-1,1,1,90,90,90)])
def test_invalid_metric_is_rejected(cell):
    with pytest.raises(ValueError):
        direct_cell_matrix(cell)


def test_density_and_composition_are_derived_from_occupied_cell():
    copper = built_in_phase("cu_fcc")
    assert copper.density_g_cm3 == pytest.approx(4*63.546*ATOMIC_MASS_UNIT_G/(3.61**3*1e-24),rel=1e-14)
    assert 8.9 < copper.density_g_cm3 < 9.1
    assert copper.elemental_mass_fractions == {"Cu":1.}
    alloy = PhaseCrystal("substitutional",(3,3,3,90,90,90),(
        AtomicSite("Fe",(0,0,0),.25),AtomicSite("Nb",(0,0,0),.75)))
    assert sum(alloy.elemental_mass_fractions.values()) == pytest.approx(1)
    expected = .25*55.845/(.25*55.845+.75*92.90637)
    assert alloy.elemental_mass_fractions["Fe"] == pytest.approx(expected)
    with pytest.raises(ValueError):
        PhaseCrystal("overoccupied",alloy.cell_A,alloy.sites+(AtomicSite("Ti",(0,0,0)),))


def test_radius_enumeration_complete_for_skew_cell_and_includes_friedel_partners():
    phase = PhaseCrystal("skew",(3.2,4.1,5.7,72,85,104),(AtomicSite("Cu",(0,0,0)),))
    limit=.7
    table = enumerate_reflections(phase,limit)
    actual = set(map(tuple,table.hkls.tolist()))
    brute = np.asarray(list(itertools.product(range(-8,9),repeat=3)))
    norm = np.linalg.norm(phase.reciprocal_vectors(brute),axis=1)
    expected=set(map(tuple,brute[(norm>0)&(norm<=limit)].tolist()))
    assert actual == expected
    assert all(tuple(-np.asarray(hkl)) in actual for hkl in actual)
    assert np.all(table.norms_A_inv<=limit*(1+1e-14))
    assert table.cell_volume_A3 == pytest.approx(phase.cell_volume_A3)


def test_radius_bound_has_no_arbitrary_max_hkl_and_honors_margin():
    phase = PhaseCrystal("large_cell",(20,20,20,90,90,90),(AtomicSite("Cu",(0,0,0)),))
    plain = enumerate_reflections(phase,.25)
    expanded = enumerate_reflections(phase,.25,candidate_margin_A_inv=.05)
    assert (5,0,0) in set(map(tuple,plain.hkls.tolist()))
    assert (6,0,0) not in set(map(tuple,plain.hkls.tolist()))
    assert (6,0,0) in set(map(tuple,expanded.hkls.tolist()))
    with pytest.raises(ValueError,match="max_candidates"):
        enumerate_reflections(phase,3,max_candidates=100)


def test_empty_reflections_and_immutable_serializable_phase():
    phase=built_in_phase("cu_fcc")
    assert len(enumerate_reflections(phase,.01).hkls)==0
    assert PhaseCrystal.from_dict(phase.to_dict()).fingerprint == phase.fingerprint
    table=enumerate_reflections(phase,1)
    for array in [table.hkls,table.G_crystal_A_inv,table.structure_factor_squared,phase.direct_basis_A]:
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("name",["cu_fcc","fe_bcc","ti_hcp","nb_bcc","skew_substitutional"])
def test_optional_independent_gemmi_structure_factor_reference(name):
    """Reference gate is runnable with Gemmi, without adding a runtime dependency.

    Gemmi performs its own reciprocal metric and full P1 atomic-basis sum. Its
    public IT92 evaluator uses float arithmetic, hence the measured tolerance.
    """
    gemmi=pytest.importorskip("gemmi")
    phase=(PhaseCrystal("skew",(3.2,4.1,5.7,72,85,104),(
        AtomicSite("Fe",(0,0,0),.4),AtomicSite("Nb",(0,0,0),.6),
        AtomicSite("Ti",(.17,.34,.71)))) if name=="skew_substitutional" else built_in_phase(name))
    structure=gemmi.SmallStructure()
    structure.cell=gemmi.UnitCell(*phase.cell_A)
    # Explicit full atomic bases must not be expanded a second time by symmetry.
    structure.spacegroup_hm="P 1"
    for index,atom in enumerate(phase.sites):
        site=gemmi.SmallStructure.Site()
        site.label=str(index)
        site.element=gemmi.Element(atom.element)
        site.fract=gemmi.Fractional(*atom.fractional_xyz)
        site.occ=atom.occupancy
        site.u_iso=0
        structure.add_site(site)
    structure.setup_cell_images()
    calculator=gemmi.StructureFactorCalculatorX(structure.cell)
    hkls=np.asarray(list(itertools.product(range(-4,5),repeat=3)))
    norms=np.linalg.norm(phase.reciprocal_vectors(hkls),axis=1)
    hkl_lists=hkls.tolist()
    reference_norm2=np.asarray([structure.cell.calculate_1_d2(hkl) for hkl in hkl_lists])
    reference_f=np.asarray([calculator.calculate_sf_from_small_structure(structure,hkl) for hkl in hkl_lists])
    assert np.allclose(norms**2,reference_norm2,atol=1e-14,rtol=1e-14)
    assert np.allclose(phase.structure_factors(hkls),reference_f,atol=3e-5,rtol=2e-6)
    for element in phase.elemental_mass_fractions:
        g=np.linspace(0,3.999,101)
        coeff=gemmi.IT92_get_exact(gemmi.Element(element),0)
        ref=np.asarray([coeff.calculate_sf(float((value/2)**2)) for value in g])
        assert np.allclose(neutral_form_factor(element,g),ref,rtol=2e-6,atol=5e-6)
