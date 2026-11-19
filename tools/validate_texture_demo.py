"""Dose-aware numerical checks of the configured finite-population texture demo.

The 8-node pixel reference uses a fresh backend and the same production
projector at higher quadrature order. It is a refinement check, not an
independently implemented projector or an orientation-population limit.
Verified basis snapshots and ordinary expectation caches avoid duplicate work.
The full 512-square references can take several minutes per backend.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from xray_generator.api import expected
from xray_generator.cli import load_requests
from xray_generator.experiment.cache import ExpectationCache
from xray_generator.experiment.provenance import file_sha256
from xray_generator.experiment.runtime import runtime_identity
from xray_generator.physics.crystallography import enumerate_reflections
from xray_generator.physics.detector import render_reciprocal_peaks
from xray_generator.physics.reciprocal_kernel import coherent_sigma_G_A_inv,ewald_sphere_integral
from xray_generator.physics.texture import orientation_ensemble
from xray_generator.solvers.contracts import readonly_array
from xray_generator.solvers.crystallographic import (
    CLASSICAL_ELECTRON_RADIUS_A, CrystallographicRateBackend,
)


def poisson_kl(reference, candidate):
    """D(Pois(reference)||Pois(candidate)), summed over native channels.

    A local series evaluates d-log1p(d) without cancellation when the means
    almost agree. None denotes infinite divergence, never a clipped likelihood.
    """
    ref, other = np.broadcast_arrays(np.asarray(reference,float),np.asarray(candidate,float))
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(other)) or np.any(ref<0) or np.any(other<0):
        raise ValueError("Poisson means must be finite and nonnegative.")
    if np.any((ref>0)&(other==0)):
        return None
    positive=ref>0
    r=ref[positive]
    c=other[positive]
    # Relative differences are safe only near equality. Computing log means
    # separately also handles positive ratios too extreme for direct division.
    small=np.abs(c-r)<1e-3*r
    delta=(c[small]-r[small])/r[small]
    value=np.empty_like(r)
    x=delta
    value[small]=x*x*(.5+x*(-1/3+x*(.25+x*(-.2+x/6))))
    value[small]*=r[small]
    value[~small]=(c[~small]-r[~small])+r[~small]*(np.log(r[~small])-np.log(c[~small]))
    result=float(np.sum(np.maximum(value,0.))+np.sum(other[~positive]))
    return result if np.isfinite(result) else None


def difference_metrics(candidate, reference, *, counts=True):
    candidate,reference=np.asarray(candidate,float),np.asarray(reference,float)
    difference=np.abs(candidate-reference)
    total=float(reference.sum())
    denominator=max(total,np.finfo(float).tiny)
    result={"units":"expected_events" if counts else "events_per_second",
        "candidate_total":float(candidate.sum()),"reference_total":total,
        "relative_total_difference":float(abs(candidate.sum()-total)/denominator),
        "absolute_L1_difference":float(difference.sum()),
        "relative_L1_difference":float(difference.sum()/denominator),
        "maximum_pixel_absolute_difference":float(difference.max()),
        "maximum_pixel_difference_over_reference_peak":float(difference.max()/max(float(reference.max()),np.finfo(float).tiny))}
    if counts:
        result["poisson_KL_reference_to_candidate_nats"]=poisson_kl(reference,candidate)
    return result


def _progress(message):
    print(message,flush=True)


def _basis(backend, path, label):
    """Reuse only snapshots matching both physical and implementation identity."""
    started=time.perf_counter()
    restored=False
    if path.is_file():
        with np.load(path,allow_pickle=False) as data:
            header=json.loads(data["header"].tobytes().decode())
            if header["backend_fingerprint"]==backend.fingerprint and header["runtime_identity"]==runtime_identity():
                rates,errors=data["basis_rates"],data["basis_errors"]
                if rates.shape!=(3,*backend.native_shape) or errors.shape!=rates.shape:
                    raise ValueError("Stored validation basis has an invalid shape.")
                if not np.all(np.isfinite(rates)) or np.any(rates<0) or not np.all(np.isfinite(errors)) or np.any(errors<0):
                    raise ValueError("Stored validation basis has invalid values.")
                object.__setattr__(backend,"_prepared",(readonly_array(rates),readonly_array(errors),tuple(header["preparation_report"])))
                restored=True
    _progress(f"{label}: {'restored matching basis' if restored else 'rendering fresh basis'}")
    rates=backend.basis_rates()
    errors=backend.basis_errors()
    if not restored:
        header={"backend_fingerprint":backend.fingerprint,"runtime_identity":runtime_identity(),
                "preparation_report":backend.preparation_report}
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,basis_rates=rates,basis_errors=errors,
            header=np.frombuffer(json.dumps(header,sort_keys=True).encode(),dtype=np.uint8))
    elapsed=time.perf_counter()-started
    _progress(f"{label}: completed in {elapsed:.1f} s")
    return {"elapsed_s":elapsed,"reused_basis_snapshot":restored,"path":str(path)}


def _cutoff_components(narrow, wide):
    """Render only added reflections, bound unchanged-reflection kernel tails.

    No full wider backend is rendered. The split is conditional on this demo's
    cell/population/geometry, not a universal reflection bound.
    """
    rows=[]
    for index,spec in enumerate(wide.phase_specs):
        report=narrow.preparation_report[index]
        sigma=float(coherent_sigma_G_A_inv(spec.coherent_domain_nm,spec.shape_factor))
        gmax=report["g_max_A_inv"]
        retained=enumerate_reflections(spec.crystal,gmax,candidate_margin_A_inv=narrow.support_sigma*sigma)
        expanded=enumerate_reflections(spec.crystal,gmax,candidate_margin_A_inv=wide.support_sigma*sigma)
        old={tuple(hkl) for hkl in retained.hkls.tolist()}
        added=np.asarray([tuple(hkl) not in old for hkl in expanded.hkls.tolist()])
        added_image=np.zeros(wide.native_shape)
        added_tail_bound=0.
        added_analytic_rate_bound=0.
        if np.any(added):
            population=orientation_ensemble(spec.texture)
            centers=np.einsum("nij,hj->nhi",population.lab_rotation_matrices,expanded.G_crystal_A_inv[added]).reshape(-1,3)
            coefficient=(wide.incident_flux_density_photons_s_um2*1e-8
                *CLASSICAL_ELECTRON_RADIUS_A**2*wide.illuminated_volume_um3*1e12/spec.crystal.cell_volume_A3**2)
            strengths=(coefficient*population.volume_weights[:,None]*expanded.structure_factor_squared[None,added]).reshape(-1)
            # Every detector ray has |G|<=gmax, so the reverse triangle
            # inequality bounds its distance from an added reciprocal peak.
            # The complete-sphere Gaussian integral outside this radius then
            # bounds added detector signal without relying on pixel quadrature.
            minimum_distance=np.maximum(0.,np.linalg.norm(centers,axis=1)-gmax)
            added_analytic_rate_bound=float(np.dot(strengths,ewald_sphere_integral(
                centers,sigma,wide.wavelength_A,outside_radius=minimum_distance)))
            _progress(f"Cutoff decomposition: {spec.crystal.phase_id}, {int(added.sum())} added signed reflections")
            projection=render_reciprocal_peaks(centers,strengths,sigma,wavelength_A=wide.wavelength_A,
                detector=wide.detector,support_sigma=wide.support_sigma,pixel_subdivision=wide.pixel_subdivision,
                polarization=wide.polarization,max_working_bytes=wide.max_working_bytes)
            added_image=projection.image
            added_tail_bound=projection.omitted_support_rate_bound
        narrow_image=narrow.basis_rates()[index]
        denominator=max(float(narrow_image.sum()),np.finfo(float).tiny)
        retained_tail_bound=report["omitted_support_rate_bound"]
        combined_bound=retained_tail_bound+added_analytic_rate_bound
        rows.append({"phase_id":spec.crystal.phase_id,
            "narrow_reflection_count":len(retained.hkls),"wide_reflection_count":len(expanded.hkls),
            "added_signed_reflections":expanded.hkls[added].tolist(),
            "added_reflection_rate_at_support12":float(added_image.sum()),
            "added_reflection_maximum_pixel_rate":float(added_image.max()),
            "added_reflection_fraction_of_retained_rate":float(added_image.sum()/denominator),
            "added_reflection_analytic_detector_rate_bound":added_analytic_rate_bound,
            "added_reflection_analytic_rate_bound_over_retained_rate":float(added_analytic_rate_bound/denominator),
            "retained_reflection_kernel_tail_rate_bound":retained_tail_bound,
            "retained_reflection_kernel_tail_relative_L1_bound":float(retained_tail_bound/denominator),
            "added_reflection_support12_tail_rate_bound":added_tail_bound,
            "combined_cutoff_L1_rate_upper_bound":combined_bound,
            "combined_cutoff_relative_L1_rate_upper_bound":float(combined_bound/denominator),
            "tail_bound_scope":"Complete outgoing sphere for enumerated peaks only; no bound beyond the wider finite reciprocal-radius table."})
    return rows


def _annular_harmonics(backend):
    row,col=np.indices(backend.native_shape,dtype=float)
    vectors,_,_=backend.detector.sample_rays(row,col,wavelength_A=backend.wavelength_A)
    g=np.linalg.norm(vectors,axis=-1)
    phi=np.arctan2(row-backend.detector.beam_center_row_col[0],col-backend.detector.beam_center_row_col[1])
    rows=[]
    for index,spec in enumerate(backend.phase_specs):
        table=enumerate_reflections(spec.crystal,1.5)
        norms=table.norms_A_inv
        first=float(norms.min())
        other=norms[norms>first+1e-8]
        gap=float(other.min()-first) if len(other) else np.inf
        sigma=float(coherent_sigma_G_A_inv(spec.coherent_domain_nm,spec.shape_factor))
        halfwidth=min(4*sigma,.45*gap)
        mask=np.abs(g-first)<halfwidth
        intensity=backend.basis_rates()[index][mask]
        signal=float(intensity.sum())
        harmonics={}
        for order in [2,4,6]:
            value=np.sum(intensity*np.exp(1j*order*phi[mask]))/max(signal,np.finfo(float).tiny)
            harmonics[str(order)]={"amplitude":float(abs(value)),"real":float(value.real),"imaginary":float(value.imag)}
        rows.append({"phase_id":spec.crystal.phase_id,"ring_G_A_inv":first,"annulus_halfwidth_A_inv":halfwidth,
            "linked_hkls":table.hkls[np.abs(norms-first)<1e-8].tolist(),
            "annular_rate_events_s":signal,"selected_native_pixels":int(mask.sum()),"harmonics":harmonics})
    return rows


def _cutoff_exposure_bound(request, mean, components):
    # Phase volume fractions are nonnegative and sum to <=1. Thus their
    # integrated coefficients sum to <= shutter-open duration, including beam
    # motion and partial material occupancy. This intentionally avoids fitting
    # coefficients from rendered images or assuming equal phase mass/volume.
    duration=(request.exposure.stop_s-request.exposure.start_s)*request.exposure.shutter_fraction
    bound=duration*max(row["combined_cutoff_L1_rate_upper_bound"] for row in components)
    # Background is an exact nonnegative floor, independent of whether a
    # fine-quadrature crystal rate slightly overestimates a true pixel mean.
    minimum=request.backend.background_rate_per_pixel*duration
    # Widening adds only nonnegative signal. For increment delta>=0,
    # D(Pois(mu+delta)||Pois(mu)) <= delta*log1p(delta/mu).
    # Sum(delta)<=bound then gives this conservative native-channel KL bound.
    kl_bound=0. if bound==0 else float(bound*np.log1p(bound/minimum)) if minimum>0 else None
    return {"exposure_id":request.exposure.exposure_id,"production_fine8_expected_events":float(np.sum(mean)),
        "L1_difference_counts_upper_bound":bound,"relative_L1_difference_upper_bound":bound/float(np.sum(mean)),
        "relative_total_difference_upper_bound":bound/float(np.sum(mean)),
        "maximum_pixel_difference_counts_upper_bound":bound,
        "poisson_KL_wider_reference_to_candidate_upper_bound_nats":kl_bound,
        "KL_baseline_floor_counts_per_pixel":minimum,
        "relative_bound_normalization":"Retained fine8 expected-event total, a quadrature estimate.",
        "propagation":"shutter-open duration * maximum per-phase L1 rate bound; uses phase-volume closure <=1"}


def run_validation(config_path, folder, cache_path):
    config,requests=load_requests(config_path)
    production=requests[0].backend
    if not isinstance(production,CrystallographicRateBackend):
        raise ValueError("A crystallographic demo configuration is required.")
    if production.pixel_subdivision!=4 or production.comparison_subdivision!=2 or production.support_sigma!=7.5:
        raise ValueError("This gate expects the declared production 4/2-node, 7.5-sigma demo profile.")
    cache=ExpectationCache(64*1024**2,disk_directory=cache_path,max_disk_bytes=64*1024**2)
    timings={"production":_basis(production,folder/"prod_basis.npz","Production 4/2")}
    prod_results=[expected(request,cache=cache) for request in requests]
    definition=production.to_dict()
    definition.update(pixel_subdivision=8,comparison_subdivision=4)
    refined=CrystallographicRateBackend.from_dict(definition)
    timings["refined"]=_basis(refined,folder/"ref_basis.npz","Refined 8/4")
    ref_results=[expected(replace(request,backend=refined),cache=cache) for request in requests]
    pixels=[]
    for prod,ref in zip(prod_results,ref_results):
        pixels.append(dict(exposure_id=prod.exposure_id,**difference_metrics(prod.total_mean,ref.total_mean)))
    definition["support_sigma"]=12.
    wide=CrystallographicRateBackend.from_dict(definition)
    cutoff_started=time.perf_counter()
    components=_cutoff_components(refined,wide)
    timings["cutoff_decomposition"]={"elapsed_s":time.perf_counter()-cutoff_started,"full_wide_backend_rendered":False}
    cutoffs=[_cutoff_exposure_bound(request,result.total_mean,components) for request,result in zip(requests,ref_results)]
    texture={"status":"descriptive_only","isotropic_comparison":"not_run; avoided another full finite-population rendering",
        "orientation_population_convergence":"not_assessed",
        "image_normalization":"none; only the dimensionless harmonic statistic divides by its own annular rate",
        "harmonic_definition":"abs(sum_pixel I(pixel)*exp(i*m*phi(pixel)) / sum_pixel I(pixel))",
        "production_annular_harmonics":_annular_harmonics(production)}
    thresholds={"relative_L1_pixel":1e-3,"poisson_KL_pixel_nats":.1,
                "relative_L1_cutoff":1e-6,"poisson_KL_cutoff_nats":.1}
    checks={"pixel_refinement_relative_L1":all(row["relative_L1_difference"]<thresholds["relative_L1_pixel"] for row in pixels),
        "pixel_refinement_dose_aware_Poisson_KL":all(row["poisson_KL_reference_to_candidate_nats"] is not None and row["poisson_KL_reference_to_candidate_nats"]<thresholds["poisson_KL_pixel_nats"] for row in pixels),
        "combined_support_and_reflection_cutoff_bound":all(row["relative_L1_difference_upper_bound"]<thresholds["relative_L1_cutoff"] for row in cutoffs),
        "cutoff_dose_aware_Poisson_KL_bound":all(row["poisson_KL_wider_reference_to_candidate_upper_bound_nats"] is not None and row["poisson_KL_wider_reference_to_candidate_upper_bound_nats"]<thresholds["poisson_KL_cutoff_nats"] for row in cutoffs),
        "fixed_reflection_kernel_tail_bound":all(row["retained_reflection_kernel_tail_relative_L1_bound"]<1e-6 for row in components),
        "added_reflection_signal":all(row["added_reflection_fraction_of_retained_rate"]<1e-6 for row in components),
        "added_reflection_analytic_bound":all(row["added_reflection_analytic_rate_bound_over_retained_rate"]<1e-6 for row in components)}
    return {"schema_version":"xray_generator.texture_demo_validation/1","status":"pass" if all(checks.values()) else "fail",
        "configuration_path":str(config_path),"configuration_sha256":file_sha256(config_path),
        "runtime_identity":runtime_identity(),"production_backend_fingerprint":production.fingerprint,
        "source_flux_density_photons_s_um2":production.incident_flux_density_photons_s_um2,
        "illuminated_volume_um3":production.illuminated_volume_um3,
        "checks":checks,"thresholds":thresholds,
        "threshold_rationale":"Pixel L1<0.1% limits integrated redistribution; Poisson KL<0.1 nat at the actual configured dose limits likelihood change. Cutoff L1<1 ppm is a demo-specific numerical regression target, not a universal physical tail bound.",
        "pixel_quadrature_refinement":{"scope":"Same finite material/orientation realization, reflection radius and 7.5-sigma support; 4 vs 8 Gauss nodes/axis. Higher order uses the same projector implementation.","per_exposure":pixels},
        "combined_support_and_reflection_cutoff":{"scope":"Conservative analytic bound for widening support/enumeration margin from 7.5 to 12 sigma. Retained-list tails use the full-sphere integral outside 7.5 sigma. Added peaks use the full-sphere integral outside max(0,|Gpeak|-gmax), independent of pixel quadrature. No full wider rendering.","per_exposure":cutoffs},
        "cutoff_decomposition":{"scope":"Separate retained-reflection kernel-tail bound and added signed-HKL detected-rate estimate plus independent analytic upper bound. Conditional on this finite wider reciprocal table; excluded higher-order reflections remain unbounded. Floating arithmetic is used, not rigorous interval arithmetic.","per_phase":components},
        "texture_diagnostic":texture,"timings":timings,"cache":cache.info(),
        "orientation_population_convergence":"not_assessed; fixed persistent finite populations reused exactly",
        "experimental_validation":"not_run; absolute instrument scale remains uncalibrated",
        "production_preparation":production.preparation_report,"refined_preparation":refined.preparation_report,
        "certification_scope":"Finite-population numerical regression for the configured demo; orientation-population and experimental validation are not certified."}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/v1/crystallographic_texture_demo.json")
    parser.add_argument("--output",type=Path,default=ROOT/"datasets/validation/texture_demo.json")
    parser.add_argument("--basis-dir",type=Path,default=ROOT/"datasets/validation/texture_demo_basis")
    parser.add_argument("--cache-dir",type=Path,default=ROOT/"datasets/texture_demo_cache")
    args=parser.parse_args(argv)
    report=run_validation(args.config,args.basis_dir,args.cache_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(report,indent=2,allow_nan=False))
    return int(report["status"]!="pass")


if __name__=="__main__":
    raise SystemExit(main())
