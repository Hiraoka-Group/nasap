import numpy as np
import pytest
from scipy.optimize import differential_evolution

from nasap.fitting import make_objective_func_from_simulating_func
from nasap.fitting.sample_data import get_a_to_b_sample

# TODO: Add more tests

def test_use_for_differential_evolution():
    # A -> B
    sample = get_a_to_b_sample()

    objective_func = make_objective_func_from_simulating_func(
        sample.t, sample.y, sample.simulating_func, sample.y0)

    result = differential_evolution(objective_func, [(-3, 3)])

    assert result.success
    np.testing.assert_allclose(result.x, sample.params.log_k, atol=1e-3)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
