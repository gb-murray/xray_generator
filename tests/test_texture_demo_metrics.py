"""Numerical likelihood diagnostics must remain valid for weak features."""
import numpy as np
import pytest
from tools.validate_texture_demo import poisson_kl


def test_poisson_kl_equal_zero_and_support_mismatch():
    assert poisson_kl([0.,1.,1e6],[0.,1.,1e6])==0.
    assert poisson_kl([0.],[3.])==3.
    assert poisson_kl([1.],[0.]) is None
    with pytest.raises(ValueError): poisson_kl([1.],[-1.])


def test_poisson_kl_tiny_positive_candidate_and_close_means():
    # Computing log1p((mu-lambda)/lambda) alone rounds the first ratio to -1.
    assert poisson_kl([1.],[1e-100])==pytest.approx(100*np.log(10.)-1.,rel=1e-14)
    epsilon=1e-8
    actual=poisson_kl([1e6],[1e6*(1+epsilon)])
    assert actual==pytest.approx(1e6*epsilon**2/2,rel=1e-7)
