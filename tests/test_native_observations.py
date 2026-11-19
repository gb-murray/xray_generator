import unittest
import numpy as np

from xray_generator.experiment.random import keyed_rng
from xray_generator.measurement.readout import apply_analysis_mask, poisson_readout
from xray_generator.preprocessing.polar import native_to_polar_bins, partition_counts


class TestNativeObservations(unittest.TestCase):
    def test_keyed_noise_order_reproduction_and_moments(self):
        expected = np.array([2., 17., 60.])
        a = poisson_readout(expected, root_seed=17, readout_id="a")
        poisson_readout(expected, root_seed=17, readout_id="unrelated")
        np.testing.assert_array_equal(a, poisson_readout(expected, root_seed=17, readout_id="a"))
        draws = poisson_readout(np.broadcast_to(expected, (60000,3)), root_seed=71,
                                readout_id="native-statistics-fixture")
        # Fixed simultaneous 7-SE mean and variance guardrails; no per-pixel fishing.
        np.testing.assert_array_less(abs(draws.mean(0)-expected), 7*np.sqrt(expected/len(draws)))
        variance_se = np.sqrt((expected + 2*expected**2)/len(draws))
        np.testing.assert_array_less(abs(draws.var(0)-expected), 7*variance_se)
        covariance = np.cov(draws, rowvar=False, ddof=0)
        for i, j in ((0,1),(0,2),(1,2)):
            self.assertLess(abs(covariance[i,j]), 7*np.sqrt(expected[i]*expected[j]/len(draws)))
        self.assertNotEqual(keyed_rng(1,"sample").integers(2**63), keyed_rng(1,"readout").integers(2**63))

    def test_exact_integer_partition_masks_and_empty_bins(self):
        raw = np.array([[2**53+1, 3], [4, 5]], dtype=np.int64)
        bins = np.array([[0,0], [1,-1]])
        normal = np.array([[1.,3.], [2.,5.]])
        valid = np.array([[True,False],[True,True]])
        result = partition_counts(raw,bins,3,normalizers=normal,valid_mask=valid)
        np.testing.assert_array_equal(result.counts,[2**53+1,4,0])
        np.testing.assert_allclose(result.coverage,[.25,1,0])
        np.testing.assert_array_equal(result.normalization,[1,2,0])
        self.assertEqual(int(result.counts.sum()),2**53+5)
        masked = apply_analysis_mask(raw,valid)
        self.assertEqual(raw[0,1],3)
        self.assertEqual(masked[0,1],0)
        all_hidden = partition_counts(raw,bins,3,valid_mask=np.zeros_like(valid))
        np.testing.assert_array_equal(all_hidden.coverage,[0,0,0])

    def test_periodic_mapping_and_count_range_validation(self):
        bins = native_to_polar_bins([1.,2.,3.,.5], [0.,2*np.pi,np.pi,0.], [1.,2.,3.], [0.,np.pi,2*np.pi])
        np.testing.assert_array_equal(bins,[0,2,3,-1])
        with self.assertRaises(ValueError):
            partition_counts(np.array([1.5]),np.array([0]),1)
        with self.assertRaises(OverflowError):
            partition_counts(np.array([np.iinfo(np.int64).max,1]),np.array([0,0]),1)
        for expected in ([float("inf")],[-1.]):
            with self.assertRaises(ValueError):
                poisson_readout(expected)


if __name__ == "__main__":
    unittest.main()
