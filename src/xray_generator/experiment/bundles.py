"""Atomic, checksum-verified fixture outputs; never overwrite corrupt results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from .provenance import canonical_json, filesystem_path


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(exposure_id,expectation_id,readout_id,root_seed):
    return hashlib.sha256(json.dumps([exposure_id,expectation_id,readout_id,int(root_seed),
                                     "native_poisson_v1"],separators=(",",":")).encode()).hexdigest()


def write_run_record(root, record):
    """Store the reconstructable compact truth/config once for all replicas."""
    directory=filesystem_path(root)/".runs"
    directory.mkdir(parents=True,exist_ok=True)
    payload=canonical_json(record).encode("utf-8")
    record_id=hashlib.sha256(payload).hexdigest()
    destination=directory/(record_id+".json")
    if destination.exists():
        if destination.read_bytes()!=payload: raise ValueError("Run-record content conflict.")
    else:
        with tempfile.NamedTemporaryFile(dir=directory,suffix=".partial",delete=False) as handle:
            temporary=Path(handle.name); handle.write(payload); handle.flush()
        try: temporary.replace(destination)
        finally:
            if temporary.exists(): temporary.unlink()
    return record_id


def _run_record(root, record_id):
    if not isinstance(record_id,str) or len(record_id)!=64 or any(c not in "0123456789abcdef" for c in record_id):
        raise ValueError("Invalid run-record identity.")
    path=filesystem_path(root)/".runs"/(record_id+".json")
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=record_id: raise ValueError("Run-record checksum failed.")
    return json.loads(payload)


def inspect_bundle(directory):
    path=filesystem_path(directory)
    manifest=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    fields={"schema_version","identity","status","profile","exposure_id","expectation_id","readout_id",
            "readout_root_seed","error_kind","calibrated_physics","checksums","run_record_id","training_truth_available"}
    if set(manifest)!=fields or manifest.get("status")!="complete" or manifest.get("schema_version")!="xray_generator.fixture_bundle/1":
        raise ValueError("Observation bundle is incomplete or has an unsupported schema.")
    if set(manifest["checksums"])!={"observable.npz","private.npz"}:
        raise ValueError("Observation bundle lacks required native arrays.")
    identity=_identity(manifest["exposure_id"],manifest["expectation_id"],manifest["readout_id"],manifest["readout_root_seed"])
    if manifest["identity"]!=identity: raise ValueError("Observation identity does not match its metadata.")
    if manifest["run_record_id"] is not None:
        record=_run_record(path.parent,manifest["run_record_id"])
        if manifest["expectation_id"] not in record["expectation_ids"] or not manifest["training_truth_available"]:
            raise ValueError("Run record does not contain this expected measurement.")
        configuration=record["configuration"]
        if manifest["exposure_id"] not in {e["exposure_id"] for e in configuration["exposures"]} or manifest["readout_root_seed"]!=configuration["readout"]["root_seed"]:
            raise ValueError("Run record does not match the exposure/readout.")
    elif manifest["training_truth_available"]:
        raise ValueError("Training truth cannot be advertised without a run record.")
    for name,checksum in manifest["checksums"].items():
        if Path(name).name!=name or not (path/name).is_file() or (path/name).stat().st_size>32*1024**2 or _digest(path/name)!=checksum:
            raise ValueError("Observation bundle is missing data or failed its checksum.")
        with zipfile.ZipFile(path/name) as archive:
            if sum(info.file_size for info in archive.infolist())>32*1024**2:
                raise ValueError("Observation bundle exceeds the supported native-array size.")
    with np.load(path/"observable.npz",allow_pickle=False) as observable, np.load(path/"private.npz",allow_pickle=False) as private:
        if set(observable.files)!={"counts"} or set(private.files)!={"expected_mean","numerical_error"}:
            raise ValueError("Observation bundle has unsupported array fields.")
        counts,mean,error=observable["counts"],private["expected_mean"],private["numerical_error"]
        if counts.dtype.kind not in "iu" or np.any(counts<0) or not 0<counts.size<=262144:
            raise ValueError("Native counts must be supported nonnegative integers.")
        if mean.shape!=counts.shape or error.shape!=counts.shape or not np.all(np.isfinite([mean,error])) or np.any(mean<0) or np.any(error<0):
            raise ValueError("Expectation/error layout or values are invalid.")
    return manifest


def write_bundle(root, observation, expected, *, root_seed, run_record_id=None):
    """Return (path,reused); orphan staging folders never count as completed data."""
    root=filesystem_path(root)
    if observation.exposure_id!=expected.exposure_id or observation.expectation_id!=expected.expectation_id:
        raise ValueError("Observation and expected measurement identities differ.")
    counts=np.asarray(observation.counts)
    if counts.shape!=expected.total_mean.shape or counts.dtype.kind not in "iu" or np.any(counts<0):
        raise ValueError("Observation counts must be nonnegative integers matching the native layout.")
    if run_record_id is not None:
        record=_run_record(root,run_record_id)
        if expected.expectation_id not in record["expectation_ids"]:
            raise ValueError("Run record does not contain the expected measurement.")
    root.mkdir(parents=True,exist_ok=True)
    identity=_identity(observation.exposure_id,observation.expectation_id,observation.readout_id,root_seed)
    target=root/identity
    if target.exists():
        manifest=inspect_bundle(target)
        if manifest["identity"]!=identity: raise ValueError("Observation identity mismatch.")
        with np.load(target/"observable.npz",allow_pickle=False) as previous:
            if not np.array_equal(previous["counts"],counts):
                raise ValueError("Existing observation differs from pinned readout reproduction.")
        with np.load(target/"private.npz",allow_pickle=False) as previous:
            if not np.array_equal(previous["expected_mean"],expected.total_mean) or not np.array_equal(previous["numerical_error"],expected.numerical_error):
                raise ValueError("Existing expectation differs from pinned reproduction.")
        return target,True
    staging=root/".staging"
    staging.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="observation-",dir=staging) as tmp:
        tmp_path=Path(tmp)
        np.savez(tmp_path/"observable.npz",counts=observation.counts)
        np.savez(tmp_path/"private.npz",expected_mean=expected.total_mean,numerical_error=expected.numerical_error)
        manifest={"schema_version":"xray_generator.fixture_bundle/1","identity":identity,
                  "status":"complete","profile":expected.profile,"exposure_id":observation.exposure_id,
                  "expectation_id":observation.expectation_id,"readout_id":observation.readout_id,
                  "readout_root_seed":int(root_seed),"error_kind":expected.numerical_error_kind,
                  "run_record_id":run_record_id,"training_truth_available":run_record_id is not None,
                  "calibrated_physics":False,"checksums":{name:_digest(tmp_path/name) for name in ["observable.npz","private.npz"]}}
        (tmp_path/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        try: tmp_path.rename(target)
        except OSError:
            if not target.exists(): raise
            existing=inspect_bundle(target)
            if existing["identity"]!=identity: raise ValueError("Concurrent output identity conflict.")
    return target,False
