import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import solve_ivp

from nasap.simulation import make_simulating_func_from_ode_rhs


def test_one_reaction():
    # A -> B
    def ode_rhs(t: float, y: npt.NDArray, k: float) -> npt.NDArray:
        return np.array([-k * y[0], k * y[0]])

    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    t = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])
    k = 1

    sol = solve_ivp(ode_rhs, (t[0], t[-1]), y0, args=(k,), dense_output=True)
    expected = sol.sol(t).T

    y = simulating_func(t, y0, k)

    assert y.shape == (len(t), len(y0))
    np.testing.assert_allclose(y, expected)


def test_two_reactions():
    # A -> B -> C
    def ode_rhs(t: float, y: npt.NDArray, k1: float, k2: float) -> npt.NDArray:
        return np.array([-k1 * y[0], k1 * y[0] - k2 * y[1], k2 * y[1]])

    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    t = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0, 0])
    k1 = 1
    k2 = 1

    sol = solve_ivp(ode_rhs, (t[0], t[-1]), y0, args=(k1, k2), dense_output=True)
    expected = sol.sol(t).T

    y = simulating_func(t, y0, k1, k2)

    assert y.shape == (len(t), len(y0))
    np.testing.assert_allclose(y, expected)


def test_reversible_reaction():
    # A <-> B
    def ode_rhs(t: float, y: npt.NDArray, k1: float, k2: float) -> npt.NDArray:
        return np.array([-k1 * y[0] + k2 * y[1], k1 * y[0] - k2 * y[1]])
    
    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    t = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])
    k1 = 1
    k2 = 1

    sol = solve_ivp(ode_rhs, (t[0], t[-1]), y0, args=(k1, k2), dense_output=True)
    expected = sol.sol(t).T

    y = simulating_func(t, y0, k1, k2)

    assert y.shape == (len(t), len(y0))
    np.testing.assert_allclose(y, expected)


if __name__ == "__main__":
    pytest.main(['-v', __file__])
