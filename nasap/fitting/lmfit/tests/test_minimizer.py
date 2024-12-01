import numpy as np
import pytest
from lmfit import Parameters

from nasap.fitting.lmfit import make_lmfit_minimizer
from nasap.fitting.sample_data import get_a_to_b_sample
from nasap.simulation import make_simulating_func_from_ode_rhs


def test_use_for_simple_minimization():
    # A -> B
    sample = get_a_to_b_sample()

    params = Parameters()
    params.add('k', value=0.0)  # Initial guess

    minimizer = make_lmfit_minimizer(
        sample.t, sample.y, sample.simulating_func, sample.y0, params)

    result = minimizer.minimize()

    assert np.isclose(result.params['k'].value, sample.params.k)


def test_use_for_differential_evolution():
    # A -> B
    sample = get_a_to_b_sample()

    params = Parameters()
    params.add('k', min=0, max=10)

    minimizer = make_lmfit_minimizer(
        sample.t, sample.y, sample.simulating_func, sample.y0, params)

    result = minimizer.minimize(method='differential_evolution')

    assert np.isclose(result.params['k'].value, sample.params.k)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
