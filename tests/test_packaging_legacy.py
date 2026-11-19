"""Frozen legacy expectations and narrow dual-namespace packaging policy."""
from __future__ import annotations

import json
from pathlib import Path
import tomllib
import unittest

import numpy as np

from src.driver_codes.Simulation_Gaussian_Broadening import run_experiment


ROOT = Path(__file__).resolve().parents[1]


class TestLegacyBaseline(unittest.TestCase):
    def test_frozen_cubic_and_hcp_expectations(self):
        record = json.loads((ROOT / "tests/fixtures/legacy_expectations.json").read_text(encoding="utf-8"))
        with np.load(ROOT / record["fixture_arrays"], allow_pickle=False) as arrays:
            for name, fixture in record["fixtures"].items():
                with self.subTest(fixture=name):
                    result = run_experiment(fixture["configuration"], save=False)
                    np.testing.assert_allclose(result["image"], arrays[name], **record["parity_tolerance"])
                    self.assertEqual(result["num_grains"], fixture["num_grains"])
                    self.assertEqual(result["num_peaks"], fixture["num_peaks"])
                    self.assertEqual(result["metadata"], fixture["metadata"])
                    self.assertGreater(result["num_peaks"], 0)

    def test_package_entries_do_not_recursively_include_new_namespace_twice(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["poetry"]
        self.assertNotIn({"include": "src"}, project["packages"])
        self.assertIn({"include": "xray_generator", "from": "src"}, project["packages"])
        self.assertEqual(project["scripts"]["xrd-generate"], "src.driver_codes.run_experiment:main")
        self.assertEqual(project["scripts"]["xray-sim"], "xray_generator.cli:main")


if __name__ == "__main__":
    unittest.main()
