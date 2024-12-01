import numpy as np
import pytest
from lmfit import Parameters

from nasap.fitting.lmfit import make_lmfit_minimizer
from nasap.fitting.sample_data import get_a_to_b_sample


def test_use_for_simple_minimization():
    # A -> B
    sample = get_a_to_b_sample()

    params = Parameters()
    params.add('log_k', value=0.0)  # Initial guess

    minimizer = make_lmfit_minimizer(
        sample.t, sample.y, sample.simulating_func, sample.y0, params)

    result = minimizer.minimize()

    np.testing.assert_allclose(
        result.params['log_k'].value, sample.params.log_k, atol=1e-3)


def test_use_for_differential_evolution():
    # A -> B
    sample = get_a_to_b_sample()

    params = Parameters()
    params.add('log_k', min=-3, max=3)

    minimizer = make_lmfit_minimizer(
        sample.t, sample.y, sample.simulating_func, sample.y0, params)

    result = minimizer.minimize(method='differential_evolution')

    np.testing.assert_allclose(
        result.params['log_k'].value, sample.params.log_k, atol=1e-3)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
