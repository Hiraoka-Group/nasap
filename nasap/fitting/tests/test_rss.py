from unittest.mock import Mock

import numpy as np
import pytest
from scipy.optimize import basinhopping, differential_evolution

from nasap.fitting import calc_simulation_rss
from nasap.fitting.sample_data import get_a_to_b_sample


def test_zero_rss():
    # A -> B
    sample = get_a_to_b_sample()

    rss = calc_simulation_rss(
        sample.t, sample.y, sample.simulating_func, sample.y0, 
        k=sample.params.k)
    
    assert rss == 0.0


def test_non_zero_rss():
    # A -> B
    sample = get_a_to_b_sample()
    k_with_error = sample.params.k + 0.1  # Introduce a small error

    rss = calc_simulation_rss(
        sample.t, sample.y, sample.simulating_func, sample.y0,
        k=k_with_error)

    assert rss > 0.0

    y_with_error = sample.simulating_func(
        sample.t, sample.y0, k=k_with_error)

    assert rss == np.sum((sample.y - y_with_error)**2)


def test_ydata_with_row_of_nan():
    mock_sim_func = Mock()
    mock_sim_func.return_value = np.array(
        [[0.0, 0.0], 
         [0.0, 0.0], 
         [0.0, 0.0]])

    tdata = np.array([0.0, 1.0, 2.0])
    ydata = np.array(
        [[0.1, 0.1], 
         [0.1, np.nan],  # Introduce a NaN value
         [0.1, 0.1]])
    
    y0 = np.array([0.0, 0.0])

    rss = calc_simulation_rss(tdata, ydata, mock_sim_func, y0, k=1)

    assert rss == 0.1**2 * 5


def test_use_for_basin_hopping():
    # A -> B
    sample = get_a_to_b_sample()

    def f(x):
        k = x[0]
        return calc_simulation_rss(
            tdata=sample.t,
            ydata=sample.y,
            simulating_func=sample.simulating_func,
            y0=sample.y0,
            k=k
        )

    result = basinhopping(f, 0.0)

    assert np.isclose(result.fun, 0.0)


def test_use_for_differential_evolution():
    # A -> B
    sample = get_a_to_b_sample()

    def f(x):
        k = x[0]
        return calc_simulation_rss(
            tdata=sample.t,
            ydata=sample.y,
            simulating_func=sample.simulating_func,
            y0=sample.y0,
            k=k
        )

    result = differential_evolution(f, bounds=[(0, 10)])

    assert np.isclose(result.fun, 0.0)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
