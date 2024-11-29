import numpy as np
import pytest
from lmfit import Parameters

from nasap.fitting.lmfit import make_lmfit_minimizer
from nasap.simulation import make_simulating_func_from_ode_rhs


def test_use_for_simple_minimization():
    # A -> B
    def ode_rhs(t, y, k):
        return np.array([-k * y[0], k * y[0]])

    tdata = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])
    k = 1

    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)
    ydata = simulating_func(tdata, y0, k)

    params = Parameters()
    params.add('k', value=0.1)

    minimizer = make_lmfit_minimizer(
        tdata, ydata, simulating_func, y0, params)

    result = minimizer.minimize()

    assert np.isclose(result.params['k'].value, k)


def test_use_for_differential_evolution():
    # A -> B
    def ode_rhs(t, y, k):
        return np.array([-k * y[0], k * y[0]])

    tdata = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])
    k = 1

    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)
    ydata = simulating_func(tdata, y0, k)

    params = Parameters()
    params.add('k', min=0, max=10)

    minimizer = make_lmfit_minimizer(
        tdata, ydata, simulating_func, y0, params)

    result = minimizer.minimize(method='differential_evolution')

    assert np.isclose(result.params['k'].value, k)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
