import math
import unittest
import warnings

import numpy as np

from xray_generator.measurement.integrate import (
    IntegrationError, IntegrationOptions, integrate_vector,
)


class TestAdaptiveIntegration(unittest.TestCase):
    def test_independent_analytic_integrals_and_small_channel(self):
        result = integrate_vector(lambda x: np.array([1e8 * x**4, 1e-5 * math.exp(x)]),
                                  0, 1, options=IntegrationOptions(atol=1e-14, rtol=1e-10))
        np.testing.assert_allclose(result.value, [2e7, 1e-5 * math.expm1(1)], rtol=1e-12)
        self.assertTrue(np.all(result.error <= 1e-14 + 1e-10 * abs(result.value)))
        self.assertTrue(result.converged)
        self.assertLessEqual(result.evaluations, 10000)
        with self.assertRaises(ValueError):
            result.value.setflags(write=True)

    def test_boundary_narrow_inclusion_preserves_integral(self):
        left, right = .50001, .50002
        result = integrate_vector(lambda x: np.array([17. if left < x < right else 0., 1.]),
                                  0, 1, breakpoints=[left, right],
                                  options=IntegrationOptions(atol=1e-13, rtol=1e-10))
        np.testing.assert_allclose(result.value, [17 * (right - left), 1.], rtol=1e-12)
        self.assertEqual(result.intervals, 3)

    def test_nontrivial_adaptation_and_error_estimate(self):
        result = integrate_vector(lambda x: math.exp(-x*x), 0, 5,
                                  options=IntegrationOptions(atol=1e-11, rtol=1e-11))
        expected = math.sqrt(math.pi) / 2 * math.erf(5)
        self.assertAlmostEqual(float(result.value), expected, places=12)
        self.assertGreater(result.intervals, 1)
        self.assertEqual(result.value.shape, ())

    def test_work_exhaustion_is_failure_not_noisy_result(self):
        with self.assertRaises(IntegrationError) as caught:
            integrate_vector(lambda x: math.sin(500*x), 0, 1,
                             options=IntegrationOptions(atol=1e-14, rtol=1e-14, max_evaluations=15))
        self.assertIsNotNone(caught.exception.partial_result)
        self.assertFalse(caught.exception.partial_result.converged)

    def test_vector_working_memory_budget_is_enforced(self):
        with self.assertRaisesRegex(IntegrationError, "max_working_bytes"):
            integrate_vector(lambda x: np.ones(1000), 0, 1,
                             options=IntegrationOptions(max_working_bytes=1024))

    def test_relative_only_tolerance_with_exact_zero_channel(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = integrate_vector(lambda x: [math.exp(-x*x), 0.], 0, 5,
                                      options=IntegrationOptions(atol=0, rtol=1e-11))
        self.assertGreater(result.intervals, 1)
        self.assertEqual(result.value[1], 0.)
        self.assertEqual(result.error[1], 0.)
        self.assertAlmostEqual(result.value[0], math.sqrt(math.pi)/2*math.erf(5), places=12)

    def test_invalid_integrands_options_and_boundaries(self):
        for kwargs in ({"atol": -1}, {"rtol": float("nan")}, {"max_evaluations": True},
                       {"max_intervals": 0}, {"atol": 0, "rtol": 0}):
            with self.assertRaises(ValueError):
                IntegrationOptions(**kwargs)
        for a, b, breaks in ((1, 0, ()), (0, 1, (2,)), (0, float("inf"), ())):
            with self.assertRaises(ValueError):
                integrate_vector(lambda x: x, a, b, breakpoints=breaks)
        with self.assertRaises(ValueError):
            integrate_vector(lambda x: [float("nan")], 0, 1)
        with self.assertRaises(ValueError):
            integrate_vector(lambda x: [x] if x < .5 else [x,x], 0, 1)
        zero = integrate_vector(lambda x: [2., 3.], 1, 1)
        np.testing.assert_array_equal(zero.value, [0, 0])


if __name__ == "__main__":
    unittest.main()
