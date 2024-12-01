import numpy as np
import pytest
from lmfit import Minimizer, Parameters

from nasap.fitting.lmfit import make_objective_func_for_lmfit_minimizer
from nasap.fitting.sample_data import get_a_to_b_sample

# TODO: Add more tests


def test_use_for_lmfit_minimizer():
    # A -> B
    sample = get_a_to_b_sample()

    objective_func = make_objective_func_for_lmfit_minimizer(
        sample.t, sample.y, sample.simulating_func, sample.y0)

    params = Parameters()
    params.add('log_k', value=0.0)  # Initial guess

    minimizer = Minimizer(objective_func, params)

    result = minimizer.minimize()

    np.testing.assert_allclose(
        result.params['log_k'].value, sample.params.log_k, atol=1e-3)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
