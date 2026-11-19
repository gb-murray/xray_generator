"""Compare bounded patches with an independent full-detector quadrature."""
import numpy as np
import pytest

from xray_generator.physics.detector import PlanarDetector, render_reciprocal_peaks


def _peak_at(detector,row,col,wavelength=.5):
    y=(col-detector.beam_center_row_col[1])*detector.pixel_pitch_mm
    z=(row-detector.beam_center_row_col[0])*detector.pixel_pitch_mm
    ray=np.array([detector.distance_mm,y,z]); ray/=np.linalg.norm(ray)
    return (ray-np.array([1.,0.,0.]))/wavelength


def _independent_full_detector(center,sigma,detector,*,subdivision=8,support=8.,wavelength=.5,midpoint=False):
    """No projector patch/sample_rays or kernel helper is called here."""
    row,col=np.indices(detector.shape,dtype=float)
    output=np.zeros(detector.shape)
    if midpoint:
        offsets=(np.arange(subdivision)+.5)/subdivision-.5
        weights=np.full(subdivision,1/subdivision)
    else:
        nodes,weights=np.polynomial.legendre.leggauss(subdivision)
        offsets=nodes/2; weights=weights/2
    for ro,rw in zip(offsets,weights):
        for co,cw in zip(offsets,weights):
            dy=(col+co-detector.beam_center_row_col[1])*detector.pixel_pitch_mm
            dz=(row+ro-detector.beam_center_row_col[0])*detector.pixel_pitch_mm
            r=np.sqrt(detector.distance_mm**2+dy*dy+dz*dz)
            qx=(detector.distance_mm/r-1)/wavelength
            qy=dy/r/wavelength; qz=dz/r/wavelength
            distance2=(qx-center[0])**2+(qy-center[1])**2+(qz-center[2])**2
            kernel=np.exp(-distance2/(2*sigma*sigma))/((2*np.pi)**1.5*sigma**3)
            kernel[distance2>(support*sigma)**2]=0.
            output+=rw*cw*kernel*detector.distance_mm*detector.pixel_pitch_mm**2/r**3
    return output


@pytest.mark.parametrize("row,col",[(31.5,63.9),(31.5,64.1),(31.5,-.7),(-.7,31.5),(64.1,31.5),(-.4,-.6),(64.,64.)])
def test_outside_centers_and_corners_match_full_detector(row,col):
    detector=PlanarDetector((64,64),.1,20.)
    center=_peak_at(detector,row,col)
    sigma=.01
    actual=render_reciprocal_peaks([center],[1.],sigma,wavelength_A=.5,detector=detector,
                                  support_sigma=8.,pixel_subdivision=4,polarization=None)
    reference=_independent_full_detector(center,sigma,detector,subdivision=4)
    assert actual.image.sum()>0
    np.testing.assert_allclose(actual.image,reference,rtol=1e-11,atol=1e-13)


def test_signal_changes_continuously_across_detector_center_boundary():
    detector=PlanarDetector((64,64),.1,20.)
    signals=[]
    for col in [63.9,64.,64.1]:
        center=_peak_at(detector,31.5,col)
        signals.append(render_reciprocal_peaks([center],[1.],.01,wavelength_A=.5,detector=detector,
                           pixel_subdivision=8,support_sigma=8.,polarization=None).image.sum())
    assert signals[0]>signals[1]>signals[2]>0
    assert signals[2]/signals[0]>.5  # legacy center rejection would give zero
    assert abs(signals[0]-2*signals[1]+signals[2])<.02*signals[1]


def test_pixel_area_quadrature_refines_for_subpixel_peak():
    detector=PlanarDetector((32,32),.1,20.)
    center=_peak_at(detector,15.21,16.34)
    reference=_independent_full_detector(center,.004,detector,subdivision=96,midpoint=True)
    errors=[]
    for n in [1,4,8]:
        value=render_reciprocal_peaks([center],[1.],.004,wavelength_A=.5,detector=detector,
                pixel_subdivision=n,support_sigma=8.,polarization=None,allow_undersampled=True).image
        errors.append(np.abs(value-reference).sum()/reference.sum())
    assert errors[-1]<errors[1]<errors[0]
    assert errors[-1]<.003


def test_strength_is_linear_and_never_renormalized_after_clipping():
    detector=PlanarDetector((32,32),.1,20.)
    center=_peak_at(detector,15.5,31.8)
    one=render_reciprocal_peaks([center],[1.],.01,wavelength_A=.5,detector=detector,polarization=None)
    three=render_reciprocal_peaks([center],[3.],.01,wavelength_A=.5,detector=detector,polarization=None)
    np.testing.assert_allclose(three.image,3*one.image,rtol=1e-14,atol=1e-14)
    assert one.omitted_support_rate_bound>=0
    with pytest.raises(ValueError): one.image.setflags(write=True)


def test_projector_rejects_unsupported_geometry_budget_and_parameters():
    with pytest.raises(ValueError): PlanarDetector((0,32))
    with pytest.raises(ValueError): render_reciprocal_peaks([[0,0,0]],[-1],.01,wavelength_A=.5)
    with pytest.raises(ValueError): render_reciprocal_peaks([[0,0,0]],[1],.01,wavelength_A=.5,pixel_subdivision=0)
    with pytest.raises(MemoryError): render_reciprocal_peaks([[0,0,0]],[1],.01,wavelength_A=.5,max_working_bytes=10)


def test_narrow_kernel_cannot_be_silently_missed_by_all_quadrature_nodes():
    detector=PlanarDetector((32,32),.1,60.)
    center=_peak_at(detector,15,15,wavelength=.5166)
    for n in [2,4,8,16,32]:
        with pytest.raises(ValueError,match="undersamples reciprocal width"):
            render_reciprocal_peaks([center],[1],1e-5,wavelength_A=.5166,detector=detector,pixel_subdivision=n)
    diagnostic=render_reciprocal_peaks([center],[1],1e-5,wavelength_A=.5166,detector=detector,
                                      pixel_subdivision=2,allow_undersampled=True)
    assert diagnostic.undersampling_waived
    assert diagnostic.minimum_width_sampling_ratio<1
