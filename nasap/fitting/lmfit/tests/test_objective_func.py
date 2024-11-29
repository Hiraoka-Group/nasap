import numpy as np
import pytest
from lmfit import Minimizer, Parameters

from nasap.fitting.lmfit import make_objective_func_for_lmfit_minimizer
from nasap.simulation import make_simulating_func_from_ode_rhs

# TODO: Add more tests


def test_use_for_lmfit_minimizer():
    # A -> B
    def ode_rhs(t, y, k):
        return np.array([-k * y[0], k * y[0]])

    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    tdata = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])
    k = 1
    ydata = simulating_func(tdata, y0, k)

    objective_func = make_objective_func_for_lmfit_minimizer(
        tdata, ydata, simulating_func, y0)

    params = Parameters()
    params.add('k', value=0.1)

    minimizer = Minimizer(objective_func, params)

    result = minimizer.minimize()

    assert np.isclose(result.params['k'].value, k)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
