import numpy as np
import pytest
from scipy.optimize import differential_evolution

from nasap.fitting import make_objective_func_from_simulating_func
from nasap.fitting.sample_data import get_a_to_b_sample

# TODO: Add more tests

def test_use_for_differential_evolution():
    # A -> B
    sample = get_a_to_b_sample()

    objective_func = make_objective_func_from_simulating_func(
        sample.t, sample.y, sample.simulating_func, sample.y0,
        unpack_args=True)

    result = differential_evolution(objective_func, [(-3, 3)])

    assert result.success
    np.testing.assert_allclose(result.x, sample.params.log_k, atol=1e-3)


def test_unpack_args():
    # Imaginary simulating function with minimum at (0, 0)
    def sim_func(t, y0, k1, k2):  # Receives unpacked arguments
        return np.zeros((len(t), len(y0))) + k1**2 + k2**2
    
    # Imaginary data
    tdata = np.array([0.0, 1.0, 2.0])
    ydata = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    y0 = np.array([0.0, 0.0])
    
    # By specifying unpack_args=True, the array of the fitting parameters,
    # i.e., np.array([k1, k2]), will be unpacked into k1 and k2
    # before passing them to sim_func
    objective_func = make_objective_func_from_simulating_func(
        tdata, ydata, sim_func, y0, unpack_args=True)

    result = differential_evolution(objective_func, [(-1, 1), (-1, 1)])

    assert result.success
    assert np.allclose(result.x, [0, 0])


def test_not_unpack_args():
    # Imaginary simulating function with minimum at (0, 0)
    def sim_func(t, y0, ks):  # Receives packed arguments
        k1 = ks[0]
        k2 = ks[1]
        return np.zeros((len(t), len(y0))) + k1**2 + k2**2
    
    # Imaginary data
    tdata = np.array([0.0, 1.0, 2.0])
    ydata = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    y0 = np.array([0.0, 0.0])
    
    # By specifying unpack_args=False, the array of the fitting parameters,
    # i.e., np.array([k1, k2]), will be passed as a single argument ks
    # to sim_func
    objective_func = make_objective_func_from_simulating_func(
        tdata, ydata, sim_func, y0, unpack_args=False)  # Do not unpack arguments

    result = differential_evolution(objective_func, [(-1, 1), (-1, 1)])

    assert result.success
    assert np.allclose(result.x, [0, 0])


if __name__ == '__main__':
    pytest.main(['-v', __file__])
