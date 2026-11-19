"""Check a built wheel in an isolated import location using the active runtime.

The active runtime must already contain the project's numerical dependencies.
No packages are installed and no source-checkout path is added to PYTHONPATH.
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import zipfile


def _check_public_paths(names):
    """Reject local plans, generated data, and machine-specific audit records."""
    private_prefixes=("datasets/","local/","research/","validation_reports/",
                      "docs/m0","docs/m3b","docs/migration_baseline", "docs/simulator_architecture",
                      "docs/wheel_validation", "tools/capture_baseline", "tools/validate_m3b",
                      "tests/test_m3b_independent")
    private=[name for name in names if name.startswith(private_prefixes) or name.endswith('/.DS_Store')]
    if private:
        raise AssertionError(f"Private/generated files included in distribution: {private}")


def check_sdist(archive_path: Path) -> dict:
    """Verify portable test assets and the public boundary of a source archive."""
    archive_path=archive_path.resolve()
    with tarfile.open(archive_path,'r:gz') as archive:
        members=archive.getmembers()
        if any(member.issym() or member.islnk() for member in members):
            raise AssertionError('Source distribution must not contain links.')
        roots={member.name.split('/')[0] for member in members}
        if len(roots)!=1: raise AssertionError('Source archive must have one package root.')
        names=[member.name.split('/',1)[1] for member in members if '/' in member.name]
        if any(name.startswith('/') or '..' in Path(name).parts for name in names):
            raise AssertionError('Source distribution contains a nonlocal path.')
        _check_public_paths(names)
        required={'README.md','LICENSE','pyproject.toml','docs/simulation.md','docs/atomic_data_provenance.md',
                  'tools/validate_physics.py','tools/validate_texture_demo.py',
                  'configs/v1/crystallographic_texture_demo.json','configs/v1/reference_scan_fixture.json',
                  'configs/continuous_odf_sputter_pvd_broad.json',
                  'tests/fixtures/legacy_expectations.json','tests/fixtures/legacy_baseline_v1.npz',
                  'src/xray_generator/physics/data/it92_subset.json',
                  'src/xray_generator/physics/data/GEMMI_LICENSE.txt'}
        if required-set(names): raise AssertionError(f'Missing source assets: {sorted(required-set(names))}')
    return {'archive':archive_path.name,'archive_sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            'archive_members':len(members),'portable_test_assets':True,'private_records_excluded':True}


def check_wheel(wheel: Path) -> dict:
    wheel = wheel.resolve()
    with TemporaryDirectory(prefix="xray-wheel-") as temporary:
        target = Path(temporary)
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            _check_public_paths(names)
            required = {
                "src/__init__.py",
                "src/classes_and_functions/orientation.py",
                "src/driver_codes/run_experiment.py",
                "src/driver_codes/materials.csv",
                "xray_generator/__init__.py",
                "xray_generator/cli.py",
                "xray_generator/physics/data/it92_subset.json",
                "xray_generator/physics/data/GEMMI_LICENSE.txt",
            }
            missing = required - set(names)
            if missing:
                raise AssertionError(f"Missing required wheel files: {sorted(missing)}")
            if any(name.startswith("src/xray_generator/") for name in names):
                raise AssertionError("New package is duplicated under the legacy namespace")
            entries = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
            if len(entries) != 1:
                raise AssertionError("Wheel must have one entry-point record")
            parser = configparser.ConfigParser()
            parser.read_string(archive.read(entries[0]).decode("utf-8"))
            expected = {
                "xrd-generate": "src.driver_codes.run_experiment:main",
                "xray-sim": "xray_generator.cli:main",
            }
            for command, value in expected.items():
                if parser["console_scripts"][command] != value:
                    raise AssertionError(f"Unexpected entry point for {command}")
            for name in names:
                destination = (target / name).resolve()
                if not destination.is_relative_to(target):
                    raise AssertionError("Wheel contains a nonlocal extraction path")
            archive.extractall(target)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        probe = """
import importlib.metadata, importlib.util, json
from importlib.resources import files
from pathlib import Path
import numpy as np
import src
import src.classes_and_functions.orientation as legacy
import xray_generator
import xray_generator.cli
from xray_generator.physics import crystallography
from xray_generator.physics.texture import TextureConfig, orientation_ensemble
root = Path.cwd().resolve()
for module in (src, legacy, xray_generator, xray_generator.cli):
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
assert importlib.util.find_spec('src.xray_generator') is None
assert xray_generator.__version__ == importlib.metadata.version('2d-xray-generator')
assets = files('xray_generator.physics').joinpath('data')
assert Path(str(assets)).resolve().is_relative_to(root)
assert 'elements' in json.loads(assets.joinpath('it92_subset.json').read_text(encoding='utf-8'))
assert len(assets.joinpath('GEMMI_LICENSE.txt').read_text(encoding='utf-8')) > 100
# Imports alone would miss incorrectly packaged scientific runtime assets.
phase = crystallography.built_in_phase('cu_fcc')
table = crystallography.enumerate_reflections(phase, .9)
assert len(table.hkls) == 26
assert any(np.array_equal(hkl, [1,1,1]) for hkl in table.hkls)
assert not any(np.array_equal(hkl, [1,0,0]) for hkl in table.hkls)
g111 = np.sqrt(3)/3.61
expected = 4*crystallography.neutral_form_factor('Cu', g111)
assert np.isclose(abs(phase.structure_factors([[1,1,1]])[0]), expected)
assert 28 < crystallography.neutral_form_factor('Cu', 0.) < 30
population = orientation_ensemble(TextureConfig(kind='fiber',num_orientations=8,seed=123))
assert population.lab_rotation_matrices.shape == (8,3,3)
assert np.isclose(population.volume_weights.sum(),1.)
print(json.dumps({'legacy_import':True,'new_import':True,'duplicate_namespace':False,
                  'physics_runtime_assets':True,'explicit_cell_reflections':len(table.hkls),
                  'persistent_orientation_population':True}))
"""
        imported = subprocess.run(
            [sys.executable, "-B", "-c", probe], cwd=target, env=env,
            capture_output=True, text=True, check=True,
        )
        commands = {}
        for command, entry in expected.items():
            module, function = entry.split(":")
            call = f"import sys; from {module} import {function}; sys.argv=[{command!r},'--help']; {function}()"
            result = subprocess.run(
                [sys.executable, "-B", "-c", call], cwd=target, env=env,
                capture_output=True, text=True, check=True,
            )
            if "usage:" not in result.stdout.lower():
                raise AssertionError(f"No help output for {command}")
            commands[command] = "pass"
        return {
            "wheel": wheel.name,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "archive_members": len(names),
            "imports": json.loads(imported.stdout), "console_entry_points": commands,
            "source_checkout_on_pythonpath": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--sdist", type=Path, help="Also check a source archive for portable tests and private-file exclusion")
    args = parser.parse_args()
    report = check_wheel(args.wheel)
    if args.sdist:
        report['source_distribution']=check_sdist(args.sdist)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
