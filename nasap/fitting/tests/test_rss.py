from unittest.mock import Mock

import numpy as np
import pytest

from nasap.fitting import calc_simulation_rss
from nasap.simulation import make_simulating_func_from_ode_rhs


def test_zero_rss():
    # A -> B
    def ode_rhs(t, y, k):
        return np.array([-k * y[0], k * y[0]])

    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    tdata = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])
    k = 1

    ydata = simulating_func(tdata, y0, k)
    params_d = {'k': k}

    rss = calc_simulation_rss(params_d, simulating_func, tdata, ydata, y0)

    assert rss == 0.0


def test_non_zero_rss():
    # A -> B
    def ode_rhs(t, y, k):
        return np.array([-k * y[0], k * y[0]])

    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    tdata = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])
    k = 1

    ydata = simulating_func(tdata, y0, k)
    ydata[0, 0] += 0.1  # Introduce a small error
    params_d = {'k': k}

    rss = calc_simulation_rss(params_d, simulating_func, tdata, ydata, y0)

    assert rss > 0.0


@pytest.mark.parametrize(
    'sim_return, ydata',
    [
        ([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        ([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], [[1.0, 0.0], [0.5, 0.5], [0.0, 1.1]]),
        ([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], [[1.0, 0.0], [0.5, 0.5], [0.1, 1.1]]),
    ]
)
def test_rss_calculation_with_mock(sim_return, ydata):
    mock_simulating_func = Mock()
    mock_simulating_func.return_value = np.array(sim_return)

    tdata = np.array([0, 1, 2])
    ydata = np.array(ydata)  # Introduce a small error
    y0 = np.array([1.0, 0.0])
    params_d = {'k': 1}

    rss = calc_simulation_rss(params_d, mock_simulating_func, tdata, ydata, y0)

    assert rss == np.sum((ydata - mock_simulating_func.return_value)**2)


if __name__ == '__main__':
    pytest.main(['-v', __file__])