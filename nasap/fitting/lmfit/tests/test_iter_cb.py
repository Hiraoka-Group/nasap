import numpy as np
import numpy.typing as npt
import pytest
from lmfit import Minimizer, Parameters

from nasap.fitting.lmfit import (IterationRecord,
                                 make_iter_cb_for_lmfit_minimizer,
                                 make_objective_func_for_lmfit_minimizer)
from nasap.simulation import make_simulating_func_from_ode_rhs


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

    iter_cb, records = make_iter_cb_for_lmfit_minimizer()
    minimizer = Minimizer(objective_func, params, iter_cb=iter_cb)

    result = minimizer.minimize()

    assert np.isclose(result.params['k'].value, k)
    assert len(records) > 0
    assert isinstance(records[0], IterationRecord)
    assert records[0].params.keys() == {'k'}
    assert records[0].iter == 0
    assert isinstance(records[0].resid, float)  # because the objective function returns a float


def test_case_where_objective_func_returns_array():
    # A -> B
    def ode_rhs(t, y, k):
        return np.array([-k * y[0], k * y[0]])

    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    tdata = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])
    k = 1
    ydata = simulating_func(tdata, y0, k)

    def objective_func(params: Parameters) -> npt.NDArray:
        k = params['k']
        ymodel = simulating_func(tdata, y0, k)
        return ymodel - ydata  # This is an array

    params = Parameters()
    params.add('k', value=0.1)

    iter_cb, records = make_iter_cb_for_lmfit_minimizer()
    minimizer = Minimizer(objective_func, params, iter_cb=iter_cb)

    result = minimizer.minimize()

    assert np.isclose(result.params['k'].value, k)
    assert len(records) > 0
    assert isinstance(records[0], IterationRecord)
    assert records[0].params.keys() == {'k'}
    assert records[0].iter == 0
    assert isinstance(records[0].resid, np.ndarray)  # because the objective function returns an array


if __name__ == '__main__':
    pytest.main(['-v', __file__])
