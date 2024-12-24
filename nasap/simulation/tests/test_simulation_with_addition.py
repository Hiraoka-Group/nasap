import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import solve_ivp

from nasap.simulation import simulate_with_addition


# A -> B
def ode_rhs(
        t: float, y: npt.NDArray, log_k_array: npt.NDArray
        ) -> npt.NDArray:
    k_array = 10.0 ** log_k_array
    k = k_array[0]
    a, b = y
    da_dt = -k * a
    db_dt = k * a
    return np.array([da_dt, db_dt])

@pytest.fixture
def t() -> npt.NDArray:
    return np.array([5, 10, 15, 20, 25, 30, 60, 120, 180, 300])

@pytest.fixture
def t0() -> float:
    return 0

@pytest.fixture
def y0() -> npt.NDArray:
    return np.array([1, 0]) * 1e-3

@pytest.fixture
def log_k_array() -> npt.NDArray:
    return np.array([-2])


def test_without_addition(t, t0, y0, log_k_array):
    sol = solve_ivp(
        ode_rhs, [t0, t[-1]], y0, t_eval=t, args=(log_k_array,),
        method='LSODA')
    expected_y = sol.y.T

    y = simulate_with_addition(
        ode_rhs, t, t0, y0, ode_rhs_args=(log_k_array,), addition=None,
        method='LSODA')
    np.testing.assert_allclose(y, expected_y, rtol=1e-3)


def test_addition(y0, log_k_array):
    t = np.array([5, 10, 15, 20, 25, 30, 60, 120, 180, 300])
    addition = {90.: np.array([1, 0]) * 1e-3}
    first_t = np.array([5, 10, 15, 20, 25, 30, 60])
    second_t = np.array([120, 180, 300])
    
    first_y0 = np.array([1, 0]) * 1e-3
    first_sol = solve_ivp(
        ode_rhs, [0, first_t[-1]], first_y0, t_eval=first_t, 
        args=(log_k_array,), method='LSODA')
    first_y = first_sol.y.T

    sol_for_y_at_90 = solve_ivp(
        ode_rhs, [0, 90], first_y0, t_eval=np.array([90]),
        args=(log_k_array,), method='LSODA')
    y_just_before_addition = sol_for_y_at_90.y.T[0]

    y_just_after_addition = y_just_before_addition + addition[90]
    second_sol = solve_ivp(
        ode_rhs, [90, second_t[-1]], y_just_after_addition, 
        t_eval=second_t, args=(log_k_array,), method='LSODA')
    second_y = second_sol.y.T

    expected_y = np.vstack([first_y, second_y])

    y = simulate_with_addition(
        ode_rhs, t, 0, y0, ode_rhs_args=(log_k_array,), addition=addition,
        method='LSODA')
    
    assert len(y) == len(expected_y)
    np.testing.assert_allclose(y, expected_y, atol=1e-6, rtol=1e-3)


def test_addition_at_t0(y0, log_k_array):
    t0 = 0
    t = np.array([5, 10])
    addition = {t0: np.array([1, 0]) * 1e-3}  # Addition at t0
    with pytest.raises(ValueError):
        simulate_with_addition(
            ode_rhs, t, t0, y0, ode_rhs_args=(log_k_array,), addition=addition,
            method='LSODA')


def test_addition_at_time_in_t(t0, y0, log_k_array):
    t = np.array([5, 10])
    addition = {10: np.array([1, 0]) * 1e-3}
    with pytest.raises(ValueError):
        simulate_with_addition(
            ode_rhs, t, t0, y0, ode_rhs_args=(log_k_array,), addition=addition,
            method='LSODA')


if __name__ == '__main__':
    pytest.main(['-vv', __file__])
