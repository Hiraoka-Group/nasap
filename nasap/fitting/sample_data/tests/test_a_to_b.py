from collections.abc import Callable

import numpy as np
import pytest

from nasap.fitting.sample_data import AToBParams, get_a_to_b_sample


def test_default_values():
    sample = get_a_to_b_sample()  # use default values
    np.testing.assert_allclose(sample.t, np.logspace(-3, 1, 10))
    assert isinstance(sample.simulating_func, Callable)
    assert sample.params == AToBParams(k=1.0)
    sim_result = sample.simulating_func(
        sample.t, np.array([1, 0]), sample.params.k)
    np.testing.assert_allclose(sim_result, sample.y)


def test_custom_values():
    t = np.logspace(-2, 1, 20)
    y0 = np.array([0.5, 0.5])
    k = 2.0
    sample = get_a_to_b_sample(t=t, y0=y0, k=k)  # use custom values
    np.testing.assert_allclose(sample.t, t)
    np.testing.assert_allclose(sample.y[0], y0)
    assert sample.params.k == k

    sim_result = sample.simulating_func(
        sample.t, sample.y[0], sample.params.k)
    np.testing.assert_allclose(sim_result, sample.y)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
