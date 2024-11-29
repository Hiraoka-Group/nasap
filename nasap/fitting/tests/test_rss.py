from unittest.mock import Mock

import numpy as np
import pytest
from scipy.optimize import basinhopping, differential_evolution

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

    rss = calc_simulation_rss(tdata, ydata, simulating_func, y0, k=k)

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

    rss = calc_simulation_rss(tdata, ydata, simulating_func, y0, k=k)

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

    rss = calc_simulation_rss(tdata, ydata, mock_simulating_func, y0, k=1)

    assert rss == np.sum((ydata - mock_simulating_func.return_value)**2)


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
    def ode_rhs(t, y, k):
        return np.array([-k * y[0], k * y[0]])

    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    tdata = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])

    def f(x):
        k = x[0]
        return calc_simulation_rss(
            tdata=np.logspace(-3, 1, 12), 
            ydata=simulating_func(tdata, y0, k),
            simulating_func=simulating_func,
            y0=y0,
            k=k
        )

    result = basinhopping(f, [0.0])

    assert result.fun == 0.0


def test_use_for_differential_evolution():
    # A -> B
    def ode_rhs(t, y, k):
        return np.array([-k * y[0], k * y[0]])

    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    tdata = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])

    def f(x):
        k = x[0]
        return calc_simulation_rss(
            tdata=np.logspace(-3, 1, 12), 
            ydata=simulating_func(tdata, y0, k),
            simulating_func=simulating_func,
            y0=y0,
            k=k
        )

    result = differential_evolution(f, bounds=[(0, 10)])

    assert result.fun == 0.0


if __name__ == '__main__':
    pytest.main(['-v', __file__])
