import numpy as np
import numpy.typing as npt
import pytest
from lmfit import Minimizer, Parameters

from nasap.fitting.lmfit import (IterationRecord,
                                 make_iter_cb_for_lmfit_minimizer,
                                 make_objective_func_for_lmfit_minimizer)
from nasap.fitting.sample_data import get_a_to_b_sample


def test_use_for_lmfit_minimizer():
    # A -> B
    sample = get_a_to_b_sample()

    objective_func = make_objective_func_for_lmfit_minimizer(
        sample.t, sample.y, sample.simulating_func, sample.y0)
    # `objective_func` returns a float

    params = Parameters()
    params.add('k', value=0.0)  # Initial guess

    iter_cb, records = make_iter_cb_for_lmfit_minimizer()
    minimizer = Minimizer(objective_func, params, iter_cb=iter_cb)

    result = minimizer.minimize()

    assert np.isclose(result.params['k'].value, sample.params.k)
    assert len(records) > 0
    assert isinstance(records[0], IterationRecord)
    assert records[0].params.keys() == {'k'}
    assert records[0].iter == 0
    # The type of `resid` should be the same as the return type of 
    # `objective_func`
    assert isinstance(records[0].resid, float)


def test_case_where_objective_func_returns_array():
    # A -> B
    sample = get_a_to_b_sample()

    def objective_func(params: Parameters) -> npt.NDArray:
        k = params['k']
        ymodel = sample.simulating_func(sample.t, sample.y0, k)
        return ymodel - sample.y  # This is an array

    params = Parameters()
    params.add('k', value=0.0)  # Initial guess

    iter_cb, records = make_iter_cb_for_lmfit_minimizer()
    minimizer = Minimizer(objective_func, params, iter_cb=iter_cb)

    result = minimizer.minimize()

    assert np.isclose(result.params['k'].value, sample.params.k)
    assert len(records) > 0
    assert isinstance(records[0], IterationRecord)
    assert records[0].params.keys() == {'k'}
    assert records[0].iter == 0
    assert isinstance(records[0].resid, np.ndarray)  # array


if __name__ == '__main__':
    pytest.main(['-v', __file__])
