from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from nasap.fitting.sample_data import AToBParams, get_a_to_b_sample


def test_default_values():
    sample = get_a_to_b_sample()  # use default values
    assert np.allclose(sample.t, np.logspace(-3, 1, 10))
    assert isinstance(sample.simulating_func, Callable)
    assert sample.params == AToBParams(k=1.0)
    sim_result = sample.simulating_func(
        sample.t, np.array([1, 0]), sample.params.k)
    assert np.allclose(sim_result, sample.y)


@given(
    tdata=arrays(
        dtype=float, shape=array_shapes(min_dims=1, max_dims=1),
        elements=st.floats(min_value=0.0, max_value=10.0)),
    y0=arrays(
        dtype=float, shape=(2,),
        elements=st.floats(min_value=0.0, max_value=10.0)),
    k=st.floats(min_value=0.0, max_value=10.0))
def test(tdata, y0, k):
    tdata.sort()
    sample = get_a_to_b_sample(
        tdata=tdata, y0=y0, k=k)  # use custom values
    assert isinstance(sample.t, np.ndarray)
    assert sample.t.ndim == 1
    n = sample.t.size
    assert isinstance(sample.y, np.ndarray)
    assert sample.y.shape == (n, 2)
    assert isinstance(sample.simulating_func, Callable)
    assert isinstance(sample.params, AToBParams)

    sim_result = sample.simulating_func(
        sample.t, sample.y[0], sample.params.k)
    assert np.allclose(sim_result, sample.y)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
