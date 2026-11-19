"""Normalization checks independent of any cropped detector image."""
import numpy as np
from scipy.integrate import quad

from xray_generator.physics.reciprocal_kernel import (
    coherent_sigma_G_A_inv, ewald_sphere_integral,
    gaussian_reciprocal_density, powder_orientation_average,
)


def test_reciprocal_kernel_has_unit_three_dimensional_mass():
    for sigma in [.002,.02,.2]:
        # Independent 1-D marginal integration; other two coordinates integrated
        # analytically. The three-dimensional unit mass follows separability.
        mass=quad(lambda x: float(gaussian_reciprocal_density([x,0.,0.],sigma))*2*np.pi*sigma**2,
                  -10*sigma,10*sigma,epsabs=1e-12)[0]
        assert abs(mass-1)<1e-12
        origin=float(gaussian_reciprocal_density([0.,0.,0.],sigma))
        assert np.isclose(origin*sigma**3,(2*np.pi)**-1.5,rtol=1e-14)


def test_coherent_width_follows_scherrer_without_intensity_rescaling():
    narrow=coherent_sigma_G_A_inv(100.)
    wide=coherent_sigma_G_A_inv(50.)
    assert np.isclose(wide/narrow,2.)
    width=wide*2*np.sqrt(2*np.log(2))
    assert np.isclose(width,.9/(50*10))


def test_fixed_volume_powder_scattering_is_domain_size_invariant():
    wavelength=.5; g=.5
    signals=[]
    for size_nm in [10.,20.,100.]:
        sigma=float(coherent_sigma_G_A_inv(size_nm))
        # Exact powder average followed by an independent Ewald-area integral:
        # dOmega = 2*pi*lambda^2*r dr. The same volume coefficient multiplies
        # each result. Finite widths remain far from zero/backscatter edges.
        value=quad(lambda r: 2*np.pi*wavelength**2*r*float(powder_orientation_average(r,g,sigma)),
                   max(0,g-12*sigma),min(2/wavelength,g+12*sigma),epsabs=1e-12)[0]
        signals.append(value)
    np.testing.assert_allclose(signals,wavelength**2/(2*g),rtol=2e-11,atol=1e-13)
    assert abs(signals[-1]/signals[0]-1)<2e-11


def test_powder_average_matches_direct_orientation_quadrature():
    g=.5; sigma=.025; r=.49
    direct=quad(lambda cosine: float(gaussian_reciprocal_density(
                    [r-g*cosine,-g*np.sqrt(max(0.,1-cosine*cosine)),0.],sigma))*.5,
                -1,1,epsabs=1e-10,points=[.95,.99])[0]
    assert np.isclose(powder_orientation_average(r,g,sigma),direct,rtol=1e-10)


def test_ewald_sphere_integral_and_tail_bound_match_direction_quadrature():
    wavelength=.5; k=1/wavelength; sigma=.03
    # Reciprocal peak centered exactly on the outgoing sphere.
    center=np.array([-.0625,np.sqrt(.25-.0625**2),0.])
    length=np.linalg.norm(center+np.array([k,0,0]))
    def scalar(u):
        distance=np.sqrt(max(0,k*k+length*length-2*k*length*u))
        return 2*np.pi*float(gaussian_reciprocal_density([distance,0,0],sigma))
    full=quad(scalar,-1,1,points=[1-.01,1-.001],epsabs=1e-11)[0]
    assert np.isclose(full,ewald_sphere_integral(center,sigma,wavelength),rtol=1e-9)
    radius=4*sigma
    boundary=(k*k+length*length-radius*radius)/(2*k*length)
    outside=quad(scalar,-1,boundary,points=[boundary-.01,boundary-.001],epsabs=1e-11)[0]
    assert np.isclose(outside,ewald_sphere_integral(center,sigma,wavelength,outside_radius=radius),rtol=1e-8)


def test_zero_radius_and_zero_center_limits_are_finite():
    assert np.isfinite(powder_orientation_average(0.,.5,.02))
    assert np.isclose(powder_orientation_average(.1,0.,.02),gaussian_reciprocal_density([.1,0,0],.02))
    assert np.isfinite(ewald_sphere_integral([-2.,0.,0.],.3,.5))
