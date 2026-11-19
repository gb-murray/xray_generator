"""Independent analytic and full-grid references for the diffraction projector.

These checks validate numerical/model consistency under the stated Gaussian,
kinematic approximation. They do not calibrate an experimental instrument.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from xray_generator.physics.detector import PlanarDetector, render_reciprocal_peaks
from xray_generator.physics.crystallography import built_in_phase, enumerate_reflections
from xray_generator.physics.reciprocal_kernel import (
    coherent_sigma_G_A_inv, gaussian_reciprocal_density, powder_orientation_average,
)
from xray_generator.physics.texture import TextureConfig, orientation_ensemble
from xray_generator.experiment.runtime import runtime_identity


def center_for_pixel(detector, wavelength_A, row, col):
    """Construct an exact Ewald point independently from detector methods."""
    beam = detector.beam_center_row_col
    if beam is None:
        beam = ((detector.shape[0]-1)/2, (detector.shape[1]-1)/2)
    ray = np.array([detector.distance_mm,
                    (col-beam[1])*detector.pixel_pitch_mm,
                    (row-beam[0])*detector.pixel_pitch_mm])
    k = 1 / wavelength_A
    return k * ray / np.linalg.norm(ray) - np.array([k, 0., 0.])


def full_grid_reference(detector, centers, strengths, sigma, *, wavelength_A, order=12):
    """Full detector Gauss-Legendre pixel quadrature, no peak support/culling.

    This deliberately does not call the production geometry, Gaussian-density,
    peak-patch, candidate-filter or polarization functions.
    """
    centers = np.asarray(centers, float)
    strengths = np.asarray(strengths, float)
    sigmas = np.broadcast_to(np.asarray(sigma, float), (len(centers),))
    nodes, weights = leggauss(order)
    beam = detector.beam_center_row_col
    if beam is None:
        beam = ((detector.shape[0]-1)/2, (detector.shape[1]-1)/2)
    row, col = np.indices(detector.shape, dtype=float)
    result = np.zeros(detector.shape)
    k = 1 / wavelength_A
    for node_y, weight_y in zip(nodes/2, weights/2):
        y = (col-beam[1]+node_y)*detector.pixel_pitch_mm
        for node_z, weight_z in zip(nodes/2, weights/2):
            z = (row-beam[0]+node_z)*detector.pixel_pitch_mm
            radius = np.sqrt(detector.distance_mm**2+y*y+z*z)
            gx = k*(detector.distance_mm/radius-1)
            gy, gz = k*y/radius, k*z/radius
            omega = detector.distance_mm*detector.pixel_pitch_mm**2/radius**3
            value = np.zeros(detector.shape)
            for center, strength, width in zip(centers, strengths, sigmas):
                squared = (gx-center[0])**2+(gy-center[1])**2+(gz-center[2])**2
                value += strength*np.exp(-squared/(2*width**2))/((2*np.pi)**1.5*width**3)
            result += weight_y*weight_z*omega*value
    return result


def powder_full_sphere_reference(g, sigma, wavelength_A):
    """Analytic Haar average then integration over the complete Ewald sphere."""
    width = np.sqrt(2)*sigma
    return wavelength_A**2/(4*g)*(2*erf(g/width)
        +erf((2/wavelength_A-g)/width)-erf((2/wavelength_A+g)/width))


def reflection_cutoff_reference():
    """Compare finite-radius tables against a wider Gaussian-tail reference.

    The Cu component deliberately aligns a {220} vector toward a detector
    corner just beyond the usual enumeration margin. Both renderings retain
    12-sigma kernel support, isolating the enumeration cutoff from pixel
    quadrature and kernel-support changes. This is a regression fixture,
    not a universal bound for all user-defined cells and populations.
    """
    wavelength=.5166
    sigma=coherent_sigma_G_A_inv(8.)
    gmax=np.sqrt(8)/3.61-7.55*sigma
    shape=(48,48)
    pitch=.2
    corner_radius=np.hypot(shape[0]*pitch/2,shape[1]*pitch/2)
    distance=corner_radius/np.tan(2*np.arcsin(gmax*wavelength/2))
    detector=PlanarDetector(shape=shape,pixel_pitch_mm=pitch,distance_mm=distance)
    target=center_for_pixel(detector,wavelength,-.5,-.5)
    target/=np.linalg.norm(target)
    original=np.array([1.,1.,0.])/np.sqrt(2)
    quaternion=np.r_[1+np.dot(original,target),np.cross(original,target)]
    quaternion/=np.linalg.norm(quaternion)
    textures=[TextureConfig(kind="component",num_orientations=64,seed=61,
                    center_quaternion=tuple(quaternion),spread_deg=.01,isotropic_fraction=.25),
              TextureConfig(kind="fiber",num_orientations=64,seed=62,
                    spread_deg=20.,isotropic_fraction=.25)]
    rows=[]
    for name,texture in zip(["cu_fcc","ti_hcp"],textures):
        phase=built_in_phase(name)
        population=orientation_ensemble(texture)
        tables=[enumerate_reflections(phase,gmax,candidate_margin_A_inv=n*sigma)
                for n in [7.5,12.]]
        images=[]
        added_signal=0.
        for table in tables:
            centers=np.einsum("nij,hj->nhi",population.lab_rotation_matrices,
                               table.G_crystal_A_inv).reshape(-1,3)
            strengths=(population.volume_weights[:,None]
                       *table.structure_factor_squared[None,:]/phase.cell_volume_A3**2).reshape(-1)
            images.append(render_reciprocal_peaks(centers,strengths,sigma,
                wavelength_A=wavelength,detector=detector,support_sigma=12.,
                pixel_subdivision=8,polarization="unpolarized").image)
            if table is tables[1]:
                retained={tuple(hkl) for hkl in tables[0].hkls}
                added=np.array([tuple(hkl) not in retained for hkl in table.hkls])
                selected=np.tile(added,len(population.volume_weights))
                added_signal=float(render_reciprocal_peaks(centers[selected],strengths[selected],sigma,
                    wavelength_A=wavelength,detector=detector,support_sigma=12.,
                    pixel_subdivision=8,polarization="unpolarized").image.sum())
        total=float(images[1].sum())
        difference=float(np.abs(images[1]-images[0]).sum())
        rows.append({"phase":name,"narrow_margin_sigma":7.5,"wide_margin_sigma":12.,
                     "render_support_sigma":12.,"reflection_counts":[len(table.hkls) for table in tables],
                     "wide_signal":total,"absolute_L1_difference":difference,
                     "relative_L1_difference":difference/max(total,np.finfo(float).tiny),
                     "added_reflection_signal":added_signal,
                     "added_reflection_fraction":added_signal/max(total,np.finfo(float).tiny)})
    return {"geometry_gmax_A_inv":gmax,"coherent_domain_nm":8.,
            "orientation_count_per_phase":64,"scope":"targeted finite-radius regression, not universal tail bound",
            "rows":rows}


def run_validation():
    wavelength = .8
    g = .45
    powder_rows = []
    for domain_nm in [8., 16., 32.]:
        sigma = coherent_sigma_G_A_inv(domain_nm)
        # Ewald differential solid angle expressed in |G| is 2*pi*lambda²*r dr.
        lower, upper = max(0.,g-12*sigma), min(2/wavelength,g+12*sigma)
        numeric = quad(lambda r: 2*np.pi*wavelength**2*r*powder_orientation_average(r,g,sigma),
                       lower,upper,epsabs=1e-11,epsrel=1e-10,points=[g])[0]
        analytic = powder_full_sphere_reference(g,sigma,wavelength)
        powder_rows.append({"domain_nm":domain_nm,"sigma_G_A_inv":float(sigma),
                            "numeric":float(numeric),"analytic":float(analytic),
                            "relative_error":float(abs(numeric/analytic-1))})

    sigma = .007
    norm = quad(lambda r: 4*np.pi*r*r*gaussian_reciprocal_density([r,0.,0.],sigma),
                0.,10*sigma,epsabs=1e-11,epsrel=1e-11)[0]

    detector = PlanarDetector(shape=(36,40),pixel_pitch_mm=.1,distance_mm=100.)
    edge_rows = []
    for col in [-.6,-.51,-.49,-.4]:
        center=center_for_pixel(detector,wavelength,17.2,col)
        image=render_reciprocal_peaks([center],[1.],.003, wavelength_A=wavelength,
            detector=detector,support_sigma=7.,pixel_subdivision=8,polarization=None).image
        reference=full_grid_reference(detector,[center],[1.],.003,wavelength_A=wavelength,order=12)
        refined_reference=full_grid_reference(detector,[center],[1.],.003,wavelength_A=wavelength,order=16)
        error=float(np.abs(image-reference).sum()/reference.sum())
        edge_rows.append({"center_col":col,"accepted_signal":float(image.sum()),
                          "reference_signal":float(reference.sum()),"relative_L1_error":error,
                          "quadrature_nodes_per_axis":[8,12,16],
                          "reference_refinement_relative_L1":float(np.abs(refined_reference-reference).sum()/refined_reference.sum())})

    # An exact-Bragg single orientation has 1/sigma integrated surface strength.
    # This is intentionally different from the fixed-volume powder invariant.
    large=PlanarDetector(shape=(96,96),pixel_pitch_mm=.1,distance_mm=100.)
    center=center_for_pixel(large,1.,47.5,57.5)
    spot_rows=[]
    for width in [.003,.006]:
        image=render_reciprocal_peaks([center],[1.],width,wavelength_A=1.,detector=large,
            support_sigma=7.,pixel_subdivision=8,polarization=None).image
        analytic=1/(np.sqrt(2*np.pi)*width)
        spot_rows.append({"sigma_G_A_inv":width,"signal":float(image.sum()),
                          "complete_sphere_analytic":float(analytic),
                          "relative_error":float(abs(image.sum()/analytic-1))})

    scaled=PlanarDetector(shape=detector.shape,pixel_pitch_mm=.2,distance_mm=200.)
    center=center_for_pixel(detector,wavelength,18.2,20.1)
    first=render_reciprocal_peaks([center],[1.],.003,wavelength_A=wavelength,
        detector=detector,pixel_subdivision=4,polarization=None).image
    second=render_reciprocal_peaks([center],[1.],.003,wavelength_A=wavelength,
        detector=scaled,pixel_subdivision=4,polarization=None).image
    scale_error=float(np.max(np.abs(first-second))/first.max())

    # Coarse/fine equality cannot certify a peak narrower than all sample gaps.
    tiny_detector=PlanarDetector(shape=(32,32),pixel_pitch_mm=.1,distance_mm=60.)
    tiny_center=center_for_pixel(tiny_detector,.5166,15.,15.)
    undersampling_rejected=False
    try:
        render_reciprocal_peaks([tiny_center],[1.],1e-5,wavelength_A=.5166,
            detector=tiny_detector,pixel_subdivision=8,polarization=None)
    except ValueError as exc:
        if "undersampl" not in str(exc).lower():
            raise
        undersampling_rejected=True

    metrics={"kernel_integral":float(norm),"powder_rows":powder_rows,
             "powder_size_signal_ratio":powder_rows[-1]["numeric"]/powder_rows[0]["numeric"],
             "edge_rows":edge_rows,"exact_bragg_spot_rows":spot_rows,
             "angular_geometry_scaling_max_relative_error":scale_error,
             "undersampling_rejected":undersampling_rejected}
    cutoff=reflection_cutoff_reference()
    metrics["reflection_cutoff_reference"]=cutoff
    checks={"normalized_reciprocal_kernel":abs(norm-1)<1e-9,
            "powder_analytic_agreement":max(x["relative_error"] for x in powder_rows)<1e-8,
            "fixed_volume_powder_size_invariance":abs(metrics["powder_size_signal_ratio"]-1)<1e-8,
            "outside_center_tails":all(x["accepted_signal"]>0 for x in edge_rows),
            "edge_full_grid_agreement":max(x["relative_L1_error"] for x in edge_rows)<5e-4,
            "edge_reference_further_refinement":max(x["reference_refinement_relative_L1"] for x in edge_rows)<1.25e-4,
            "single_orientation_strength":max(x["relative_error"] for x in spot_rows)<5e-4,
            "geometry_scaling":scale_error<1e-12,
            "unresolved_peak_rejected":undersampling_rejected,
            "reflection_cutoff_fixture":max(max(row["relative_L1_difference"],row["added_reflection_fraction"])
                                             for row in cutoff["rows"])<1e-7,
            "reflection_cutoff_fixture_nonvacuous":any(row["reflection_counts"][1]>row["reflection_counts"][0]
                                                     and row["added_reflection_signal"]>0 for row in cutoff["rows"])}
    return {"schema_version":"xray_generator.physics_validation/1",
            "scope":"normalized Gaussian kinematic kernel and pixel projector; numerical/analytic fixtures",
            "runtime_identity":runtime_identity(),
            "status":"pass" if all(checks.values()) else "fail","checks":checks,"metrics":metrics,
            "physical_assumptions":["Incoherent grains/kinematic elastic scattering",
                "Isotropic Gaussian reciprocal peak envelope with width proportional to inverse coherent domain size",
                "No experimental absorption, source spectrum or detector efficiency calibration established"],
            "experimental_validation":"not_run",
            "references":["https://journals.iucr.org/a/issues/2025/04/00/ae5156/index.html",
                          "https://journals.iucr.org/m/issues/2014/01/00/cw5001/"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"datasets/validation/physics.json")
    args=parser.parse_args()
    report=run_validation()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return int(report["status"]!="pass")


if __name__=="__main__": raise SystemExit(main())
