"""Reference and crystallographic acquisitions; historical xrd-generate is retained."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .api import Exposure, SimulationRequest, expectation_dependencies, expected, observe, validate
from .experiment.bundles import inspect_bundle, write_bundle, write_run_record
from .experiment.cache import ExpectationCache
from .experiment.provenance import filesystem_path, fingerprint
from .experiment.runtime import runtime_identity
from .material import ProceduralMaterial
from .measurement.integrate import IntegrationOptions
from .model import MaterialRecipe
from .solvers.reference import ReferenceRateBackend, synthetic_ring_templates
from .solvers.crystallographic import CrystallographicRateBackend


def load_requests(path):
    config=json.loads(Path(path).read_text(encoding="utf-8"))
    allowed={"schema_version","profile","material","reference","crystallography","exposures","integration",
             "max_total_evaluations","cache_memory_bytes","readout"}
    if set(config)-allowed: raise ValueError(f"Unsupported configuration fields: {sorted(set(config)-allowed)}")
    schema=config.get("schema_version")
    if schema not in {"xray_generator.fixture_experiment/1","xray_generator.crystallographic_experiment/1"}:
        raise ValueError("Unsupported schema; use an implemented experiment schema.")
    material=ProceduralMaterial(MaterialRecipe.from_dict(config["material"]))
    if schema=="xray_generator.crystallographic_experiment/1":
        if config.get("profile")!="static_kinematic_gaussian_v1" or "reference" in config:
            raise ValueError("A crystallographic request requires its explicit profile and no reference-rate override.")
        backend=CrystallographicRateBackend.from_dict(config["crystallography"])
    else:
        if config.get("profile")!="synthetic_reference_fixture_v1" or "crystallography" in config:
            raise ValueError("Reference fixtures require their explicit profile and reference rates.")
        ref=dict(config["reference"])
        if set(ref)-{"component_ids","component_rates","synthetic_ring_size","modality","background_included"}:
            raise ValueError("Unsupported reference fields.")
        if ("component_rates" in ref)==("synthetic_ring_size" in ref):
            raise ValueError("Supply either explicit native rates or the synthetic ring fixture size.")
        rates=ref.get("component_rates")
        if rates is None: rates=synthetic_ring_templates(ref["synthetic_ring_size"])
        backend=ReferenceRateBackend(rates,tuple(ref["component_ids"]),ref.get("modality","xrd"),ref.get("background_included",True))
    options=IntegrationOptions(**config.get("integration",{}))
    requests=[SimulationRequest(material,backend,Exposure(**e),options,config.get("max_total_evaluations",100000)) for e in config["exposures"]]
    if not requests or len({r.exposure.exposure_id for r in requests})!=len(requests):
        raise ValueError("Nonempty uniquely identified exposures are required.")
    for request in requests: validate(request)
    readout=config.get("readout",{})
    if set(readout)-{"root_seed","replicate_ids"}: raise ValueError("Unsupported readout parameters.")
    seed=readout.get("root_seed",17)
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:
        raise ValueError("readout.root_seed must be a nonnegative integer.")
    ids=readout.get("replicate_ids",["0"])
    if not isinstance(ids,list) or not ids or any(isinstance(v,bool) or not isinstance(v,(str,int)) or str(v)=="" for v in ids):
        raise ValueError("readout.replicate_ids must be a nonempty list of string/integer identities.")
    if len({str(v) for v in ids})!=len(ids):
        raise ValueError("Readout identities must be distinct after string normalization.")
    memory=config.get("cache_memory_bytes",8*1024**2)
    if isinstance(memory,bool) or not isinstance(memory,int) or memory<0:
        raise ValueError("cache_memory_bytes must be a nonnegative integer.")
    return config,requests


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ["validate","expected","observe"]:
        command=sub.add_parser(name)
        command.add_argument("config",type=Path)
        if name!="validate":
            command.add_argument("--output",type=Path,required=True)
            command.add_argument("--cache-dir",type=Path)
            command.add_argument("--cache-disk-bytes",type=int,default=16*1024**2)
    gates=sub.add_parser("gates")
    gates.add_argument("--output",type=Path)
    inspect=sub.add_parser("inspect")
    inspect.add_argument("bundle",type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=="inspect":
            print(json.dumps(inspect_bundle(args.bundle),indent=2)); return 0
        if args.command=="gates":
            from .diagnostics.gates import run_development_gates
            report=run_development_gates()
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
            print(json.dumps(report,indent=2))
            return int(any(g["status"]=="fail" for g in report["gates"]))
        config,requests=load_requests(args.config)
        if args.command=="validate":
            print(json.dumps({"valid":True,"exposures":len(requests),"capabilities":validate(requests[0])},indent=2)); return 0
        args.output=filesystem_path(args.output)
        args.output.mkdir(parents=True,exist_ok=True)
        normalized=dict(config)
        normalized["material"]=requests[0].material.recipe.to_dict()
        normalized["exposures"]=[asdict(r.exposure) for r in requests]
        normalized["integration"]=asdict(requests[0].accuracy)
        if isinstance(requests[0].backend,CrystallographicRateBackend):
            normalized["crystallography"]=requests[0].backend.to_dict()
        normalized["max_total_evaluations"]=requests[0].max_total_evaluations
        normalized["readout"]={"root_seed":config.get("readout",{}).get("root_seed",17),
                                "replicate_ids":config.get("readout",{}).get("replicate_ids",["0"])}
        run_record_id=write_run_record(args.output,{"schema_version":"xray_generator.fixture_run/1",
            "configuration":normalized,"runtime_identity":runtime_identity(),
            "expectation_ids":[fingerprint(expectation_dependencies(r)) for r in requests],
            "calibrated_physics":False})
        cache=ExpectationCache(config.get("cache_memory_bytes",8*1024**2),
                               disk_directory=args.cache_dir,max_disk_bytes=args.cache_disk_bytes if args.cache_dir else 0)
        output=[]
        for request in requests:
            if args.command=="expected":
                result=expected(request,cache=cache)
                destination=args.output/(result.expectation_id+".npz")
                if destination.exists():
                    with np.load(destination,allow_pickle=False) as prior:
                        if not np.array_equal(prior["total_mean"],result.total_mean) or not np.array_equal(prior["numerical_error"],result.numerical_error):
                            raise ValueError("Existing expectation does not match regenerated content.")
                else:
                    # Cache already offers atomic persistence when requested.
                    from tempfile import NamedTemporaryFile
                    with NamedTemporaryFile(dir=args.output,suffix=".partial",delete=False) as handle:
                        temporary=Path(handle.name)
                        np.savez(handle,total_mean=result.total_mean,numerical_error=result.numerical_error)
                    try: temporary.replace(destination)
                    finally:
                        if temporary.exists(): temporary.unlink()
                output.append({"exposure_id":result.exposure_id,"expectation_id":result.expectation_id,"path":str(destination),"cache_hit":result.cache_hit,"events":float(result.total_mean.sum())})
            else:
                seed=config.get("readout",{}).get("root_seed",17)
                for replica in config.get("readout",{}).get("replicate_ids",["0"]):
                    observation,result=observe(request,root_seed=seed,readout_id=replica,cache=cache)
                    destination,reused=write_bundle(args.output,observation,result,root_seed=seed,run_record_id=run_record_id)
                    output.append({"exposure_id":result.exposure_id,"readout_id":str(replica),"path":str(destination),"cache_hit":result.cache_hit,"reused":reused,"counts":int(observation.counts.sum())})
        summary={"profile":requests[0].backend.profile,"calibrated_physics":False,
                 "run_record_id":run_record_id,"outputs":output,"cache":cache.info()}
        if isinstance(requests[0].backend,CrystallographicRateBackend) and requests[0].backend._prepared is not None:
            summary["crystallographic_preparation"]=requests[0].backend.preparation_report
        print(json.dumps(summary,indent=2))
        return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        parser.exit(2,f"xray-sim: {exc}\n")


if __name__=="__main__":
    raise SystemExit(main())
