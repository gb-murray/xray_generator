"""Independent numerical geometry and normalization references for diffraction."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

from tools.validate_physics import center_for_pixel, full_grid_reference, run_validation
from xray_generator.physics.detector import PlanarDetector, render_reciprocal_peaks
from xray_generator.physics.reciprocal_kernel import ewald_sphere_integral, powder_orientation_average


def test_independent_kernel_projector_references():
    report=run_validation()
    assert report["status"]=="pass", report["metrics"]


def test_powder_density_matches_direct_orientation_integration():
    # Independently integrate over the cosine of a uniformly oriented G vector.
    mu,w=leggauss(128)
    g,sigma=.4,.04
    for radius in [.0,.25,.4,.55]:
        direct=.5*np.sum(w*np.exp(-(radius**2+g**2-2*radius*g*mu)/(2*sigma**2)))
        direct/=((2*np.pi)**1.5*sigma**3)
        np.testing.assert_allclose(powder_orientation_average(radius,g,sigma),direct,rtol=2e-10,atol=1e-18)


def test_ewald_support_omission_matches_direct_distance_integration():
    # Distance from reciprocal peak center to a point on outgoing sphere.
    # dOmega = 2*pi*s/(k*|k_in+G|) ds, derived without production geometry.
    wavelength=.8
    k=1/wavelength
    for center,sigma in [(np.array([.05,.2,.1]),.04),
                         (np.array([-.2,.5,.1]),.08)]:
        a=np.linalg.norm(center+[k,0.,0.])
        minimum,maximum=abs(k-a),k+a
        for radius in [0.,minimum+.5*sigma,minimum+3*sigma]:
            lower=max(minimum,radius)
            direct=quad(lambda s: 2*np.pi*s/(k*a)*np.exp(-s*s/(2*sigma*sigma))
                        /((2*np.pi)**1.5*sigma**3),lower,maximum,
                        epsabs=1e-12,epsrel=1e-11)[0]
            actual=ewald_sphere_integral(center,sigma,wavelength,outside_radius=radius)
            np.testing.assert_allclose(actual,direct,rtol=1e-9,atol=1e-13)


def test_offshell_and_shifted_high_angle_detector_match_full_grid():
    # An off-center beam and a close detector exercise the conservative cap
    # bounds far from the small-angle geometry used by the demo. Centers are
    # displaced along outgoing rays, including beyond the retained support.
    wavelength=.8
    detector=PlanarDetector(shape=(18,20),pixel_pitch_mm=1.,distance_mm=12.,
                            beam_center_row_col=(-4.,5.))
    k=1/wavelength
    centers=[]
    for row,col,radial_offset in [(8.,-.6,.025),(17.6,18.,-.04),
                                  (4.,12.,.19),(6.,8.,.0)]:
        ewald=center_for_pixel(detector,wavelength,row,col)
        outgoing=ewald+[k,0.,0.]
        centers.append(ewald+radial_offset*outgoing/np.linalg.norm(outgoing))
    strengths=np.array([1.,.7,3.,.2])
    width=.025
    result=render_reciprocal_peaks(centers,strengths,width,
        wavelength_A=wavelength,detector=detector,support_sigma=7.,
        pixel_subdivision=16,polarization=None)
    reference=full_grid_reference(detector,centers,strengths,width,
                                  wavelength_A=wavelength,order=24)
    discrepancy=np.abs(result.image-reference).sum()
    assert discrepancy <= 2*result.omitted_support_rate_bound+1e-8*reference.sum()
