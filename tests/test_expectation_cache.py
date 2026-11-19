from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from xray_generator.experiment.cache import ExpectationCache
from xray_generator.experiment.provenance import expectation_key, fingerprint, filesystem_path


class TestExpectationCache(unittest.TestCase):
    def test_dependencies_and_error_certificate_gate_reuse(self):
        dependency = {"source": "hash-a", "recipe": {"seed": 1}, "pose": [0,0,0], "accuracy": 1e-6}
        key = expectation_key(dependency)
        self.assertEqual(key, expectation_key(dict(reversed(list(dependency.items())))))
        self.assertNotEqual(key, expectation_key({**dependency, "source": "hash-b"}))
        self.assertNotEqual(key, expectation_key({**dependency, "accuracy": 1e-8}))
        cache = ExpectationCache(4096)
        source = np.array([2., 3.])
        cache.put(key, source, error=[1e-6, 2e-6], metadata={"error_kind": "estimate", "owners": ["physics"]})
        source[0] = 1000
        got = cache.get(key, max_error=2e-6)
        np.testing.assert_array_equal(got.values, [2,3])
        self.assertIsNone(cache.get(key, max_error=1e-6))
        with self.assertRaises(ValueError):
            got.values.setflags(write=True)
        with self.assertRaises(TypeError):
            got.metadata["owners"] = ()
        self.assertEqual(got.metadata["owners"], ("physics",))

    def test_memory_cap_and_eviction_do_not_change_values(self):
        cache = ExpectationCache(250)
        values = np.arange(4.)
        for i in range(4):
            cache.put(fingerprint(i), values + i, error=0)
            self.assertLessEqual(cache.bytes_used, cache.max_memory_bytes)
        self.assertGreater(cache.evictions, 0)
        self.assertIsNone(cache.get(fingerprint(0)))
        np.testing.assert_array_equal(cache.get(fingerprint(3)).values, values+3)
        oversized = ExpectationCache(1)
        self.assertFalse(oversized.put(fingerprint("big"), values, error=0))
        self.assertEqual(oversized.bytes_used, 0)

    def test_atomic_disk_restart_integrity_and_caps(self):
        with TemporaryDirectory() as folder:
            key = fingerprint("disk-case")
            cache = ExpectationCache(0, disk_directory=folder, max_disk_bytes=8192)
            self.assertTrue(cache.put(key, [1.,2.,3.], error=[.1,.2,.3], metadata={"nested": {"ids": [1,2]}}))
            restarted = ExpectationCache(0, disk_directory=folder, max_disk_bytes=8192)
            got = restarted.get(key, max_error=.3)
            np.testing.assert_array_equal(got.values, [1,2,3])
            self.assertEqual(got.metadata["nested"]["ids"], (1,2))
            self.assertEqual(restarted.disk_hits, 1)
            self.assertEqual(list(Path(folder).glob("*.partial")), [])
            path = Path(folder)/f"expectation-{key}.npz"
            path.write_bytes(b"interrupted-or-corrupt")
            self.assertIsNone(restarted.get(key))
            self.assertEqual(restarted.invalidations, 1)
            # An unrelated file must never be included in eviction/deletion.
            unrelated = Path(folder)/"user-notes.txt"
            unrelated.write_text("retain me")
            for i in range(10):
                cache.put(fingerprint(i), np.arange(20.)+i, error=0)
            self.assertLessEqual(cache.disk_bytes_used, 8192)
            self.assertEqual(unrelated.read_text(), "retain me")

    def test_invalid_values_cannot_become_training_examples(self):
        cache = ExpectationCache(1000)
        key = fingerprint("invalid")
        for values, error in (([-1],0), ([float("nan")],0), ([1],-1), ([1],[1,2])):
            with self.assertRaises(ValueError):
                cache.put(key, values, error=error)
        with self.assertRaises(ValueError):
            cache.get("../../other-file")
        with self.assertRaises(ValueError):
            fingerprint({"value": float("nan")})

    def test_long_directory_atomic_disk_round_trip(self):
        with TemporaryDirectory() as folder:
            root = filesystem_path(folder)
            directory = root / ("material-" + "m" * 90) / ("expectation-" + "e" * 90)
            try:
                key = fingerprint("long-path-roundtrip")
                cache = ExpectationCache(0, disk_directory=directory, max_disk_bytes=8192)
                self.assertGreater(len(str(directory / f"expectation-{key}.npz")), 260)
                self.assertTrue(cache.put(key, [3., 7.], error=[1e-8,2e-8]))
                cold = ExpectationCache(0, disk_directory=directory, max_disk_bytes=8192)
                np.testing.assert_array_equal(cold.get(key).values, [3,7])
                self.assertEqual(cold.disk_hits, 1)
            finally:
                # Teardown must use the same extended path convention on Windows.
                shutil.rmtree(root, ignore_errors=False)


if __name__ == "__main__":
    unittest.main()
