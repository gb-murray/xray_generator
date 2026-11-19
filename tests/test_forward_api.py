"""Independent analytical/dense references and durable end-to-end fixtures."""
from dataclasses import replace
import json

import numpy as np
import pytest

from xray_generator.api import Exposure, ResourceLimitError, expected, observe
from xray_generator.cli import load_requests, main
from xray_generator.diagnostics.gates import dense_midpoint_reference, make_boundary_request, poisson_kl
from xray_generator.experiment.bundles import inspect_bundle, write_bundle
from xray_generator.experiment.cache import ExpectationCache
from xray_generator.measurement.integrate import IntegrationOptions
from xray_generator.solvers.toy_annular import AngularProfile, PilotGeometry, render_expected
from xray_generator.preprocessing.polar import native_to_polar_bins, partition_counts
from xray_generator.measurement.readout import poisson_readout


def test_stationary_motion_and_partial_illumination_analytic():
    request=make_boundary_request()
    moving=expected(request)
    np.testing.assert_allclose(moving.total_mean,[6.,11.,.15],rtol=0,atol=1e-11)
    pure=expected(replace(request,exposure=Exposure("pure",0,.2,-1.,0.,.1)))
    np.testing.assert_allclose(pure.total_mean,request.backend.component_rates[0]*.2,rtol=0,atol=1e-11)
    half=expected(replace(request,exposure=Exposure("edge",0,1.,-2.,0.,2.)))
    np.testing.assert_allclose(half.total_mean,request.backend.component_rates[0]/2,rtol=0,atol=1e-11)
    off=expected(replace(request,exposure=Exposure("shutter",0,1.,shutter_fraction=0)))
    assert np.array_equal(off.total_mean,np.zeros(3))


def test_high_resolution_reference_and_dose_sensitive_kl():
    request=make_boundary_request()
    fast=expected(request)
    coarse=dense_midpoint_reference(request,space_nodes=128,time_nodes=32)
    fine=dense_midpoint_reference(request,space_nodes=512,time_nodes=128)
    np.testing.assert_allclose(coarse,fine,rtol=0,atol=1e-11)
    for dose in [.125,1.,8.]:
        assert poisson_kl(dose*fine,dose*fast.total_mean)<.01
    assert poisson_kl([1.],[0.])==float("inf")


def test_expectation_cache_noise_and_acquisition_identity_are_separate(tmp_path):
    request=make_boundary_request()
    cache=ExpectationCache(100000,disk_directory=tmp_path,max_disk_bytes=100000)
    one,mean1=observe(request,root_seed=97,readout_id="first",cache=cache)
    repeat,mean2=observe(request,root_seed=97,readout_id="first",cache=cache)
    assert mean2.cache_hit and np.array_equal(one.counts,repeat.counts)
    changed=replace(request,exposure=replace(request.exposure,exposure_id="independent-exposure"))
    two,mean3=observe(changed,root_seed=97,readout_id="first",cache=cache)
    assert mean3.cache_hit and mean1.expectation_id==mean3.expectation_id
    assert not np.array_equal(one.counts,two.counts)
    reloaded=ExpectationCache(0,disk_directory=tmp_path,max_disk_bytes=100000)
    restored=expected(request,cache=reloaded)
    assert restored.cache_hit
    np.testing.assert_array_equal(restored.total_mean,mean1.total_mean)
    tighter=replace(request,accuracy=IntegrationOptions(atol=1e-10,rtol=1e-9))
    assert not expected(tighter,cache=reloaded).cache_hit


def test_total_work_limit_rejects_example():
    with pytest.raises(ResourceLimitError):
        expected(replace(make_boundary_request(),max_total_evaluations=10))


def test_bundle_resume_orphan_and_corruption(tmp_path):
    request=make_boundary_request()
    observation,mean=observe(request,root_seed=12)
    orphan=tmp_path/".staging"/"abandoned"
    orphan.mkdir(parents=True)
    (orphan/"observable.npz.partial").write_bytes(b"incomplete")
    path,reused=write_bundle(tmp_path,observation,mean,root_seed=12)
    assert not reused and inspect_bundle(path)["status"]=="complete"
    assert write_bundle(tmp_path,observation,mean,root_seed=12)==(path,True)
    (path/"observable.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError,match="checksum"):
        write_bundle(tmp_path,observation,mean,root_seed=12)


def test_controlled_renderer_native_budget_and_polar_counts():
    result=render_expected(components=("A","C"),weights=(.3,.7),
        profiles=(AngularProfile(),AngularProfile("antipodal_equal_von_mises",.7,16.,.2)))
    assert abs(result["signal_mean"].sum()-1024)<1e-9
    assert abs(result["background_mean"].sum()-1024)<1e-9
    counts=poisson_readout(result["total_mean"],root_seed=701)
    bins=native_to_polar_bins(result["q_A_inv"],result["chi_rad"],np.linspace(1.,4.5,257),np.linspace(0.,2*np.pi,129))
    polar=partition_counts(counts,bins,256*128,normalizers=result["normalizer"])
    assert int(polar.counts.sum())==int(counts.sum())
    valid=np.ones(counts.shape,dtype=bool); valid[:,240:260]=False
    masked=partition_counts(counts,bins,256*128,normalizers=result["normalizer"],valid_mask=valid)
    assert int(masked.counts.sum())==int(counts[valid].sum())
    assert np.all(np.isfinite(masked.coverage))


def test_cli_observe_and_resume(tmp_path,capsys):
    from pathlib import Path
    config=Path(__file__).parents[1]/"configs"/"v1"/"reference_scan_fixture.json"
    cfg,requests=load_requests(config)
    assert len(requests)==2
    assert main(["validate",str(config)])==0
    assert json.loads(capsys.readouterr().out)["valid"]
    arguments=["observe",str(config),"--output",str(tmp_path/"observations"),"--cache-dir",str(tmp_path/"cache")]
    assert main(arguments)==0
    first=json.loads(capsys.readouterr().out)
    assert len(first["outputs"])==4 and not first["calibrated_physics"]
    assert len(first["run_record_id"])==64
    from pathlib import Path
    bundle=Path(first["outputs"][0]["path"])
    manifest=inspect_bundle(bundle)
    assert manifest["training_truth_available"]
    record=json.loads((bundle.parent/".runs"/(manifest["run_record_id"]+".json")).read_text(encoding="utf-8"))
    from xray_generator.model import MaterialRecipe
    assert MaterialRecipe.from_dict(record["configuration"]["material"]).fingerprint==requests[0].material.recipe.fingerprint
    assert main(arguments)==0
    again=json.loads(capsys.readouterr().out)
    assert all(item["reused"] for item in again["outputs"])
    assert all(item["cache_hit"] for item in again["outputs"])
    record_path=bundle.parent/".runs"/(manifest["run_record_id"]+".json")
    record_path.write_bytes(b"corrupted run record")
    with pytest.raises(ValueError,match="Run-record checksum"):
        inspect_bundle(bundle)


@pytest.mark.parametrize("readout",[
    {"replicate_ids":[]},{"replicate_ids":"abc"},{"replicate_ids":[1,"1"]},
    {"replicate_ids":[True]},{"root_seed":-1},{"root_seed":1.5},{"root_seed":True},
])
def test_cli_rejects_invalid_noise_identity_before_generation(tmp_path,readout):
    from pathlib import Path
    source=Path(__file__).parents[1]/"configs"/"v1"/"reference_scan_fixture.json"
    config=json.loads(source.read_text(encoding="utf-8"))
    config["readout"]=readout
    target=tmp_path/"invalid.json"
    target.write_text(json.dumps(config),encoding="utf-8")
    with pytest.raises(ValueError): load_requests(target)
