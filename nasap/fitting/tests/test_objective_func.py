import numpy as np
import pytest
from scipy.optimize import differential_evolution

from nasap.fitting import make_objective_func_from_simulating_func
from nasap.simulation import make_simulating_func_from_ode_rhs

# TODO: Add more tests


def test_use_for_differential_evolution():
    # A -> B
    def ode_rhs(t, y, k):
        return np.array([-k * y[0], k * y[0]])

    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    tdata = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])
    k = 1
    ydata = simulating_func(tdata, y0, k)

    objective_func = make_objective_func_from_simulating_func(
        tdata, ydata, simulating_func, y0)

    result = differential_evolution(objective_func, [(0, 10)])

    assert np.isclose(result.fun, 0.0)
    assert np.isclose(result.x, k)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
