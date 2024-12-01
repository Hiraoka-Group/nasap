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
        log_k=sample.params.log_k)  # Use the correct value
    
    np.testing.assert_allclose(rss, 0.0)


def test_non_zero_rss():
    # A -> B
    sample = get_a_to_b_sample()
    log_k_with_error = sample.params.log_k + 0.1  # Introduce a small error

    rss = calc_simulation_rss(
        sample.t, sample.y, sample.simulating_func, sample.y0,
        log_k=log_k_with_error)  # Use the incorrect value

    assert rss > 0.0

    y_with_error = sample.simulating_func(
        sample.t, sample.y0, log_k=log_k_with_error)

    np.testing.assert_allclose(rss, np.sum((sample.y - y_with_error)**2))


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

    np.testing.assert_allclose(rss, 0.1**2 * 5)


def test_use_for_basin_hopping():
    # A -> B
    sample = get_a_to_b_sample()

    def f(x):
        log_k = x[0]
        return calc_simulation_rss(
            tdata=sample.t,
            ydata=sample.y,
            simulating_func=sample.simulating_func,
            y0=sample.y0,
            log_k=log_k
        )

    result = basinhopping(f, 0.0)

    assert result.success
    # We expect three decimal places of accuracy of log_k
    # i.e. abs(result.x - sample.params.log_k) < 0.001
    np.testing.assert_allclose(result.x, sample.params.log_k, atol=1e-3)


def test_use_for_differential_evolution():
    # A -> B
    sample = get_a_to_b_sample()

    def f(x):
        log_k = x[0]
        return calc_simulation_rss(
            tdata=sample.t,
            ydata=sample.y,
            simulating_func=sample.simulating_func,
            y0=sample.y0,
            log_k=log_k
        )

    result = differential_evolution(f, bounds=[(-3, 3)])

    assert result.success
    np.testing.assert_allclose(result.x, sample.params.log_k, atol=1e-3)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
