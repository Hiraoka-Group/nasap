from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import pytest

from nasap.fitting.sample_data import SampleData


def test_init() -> None:
    class Params(NamedTuple):
        k: float
    
    sim_func = lambda t, y0, k: np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    
    sample = SampleData(
        tdata=[0.0, 1.0, 2.0],
        ydata=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        simulating_func=sim_func,
        params=Params(k=1.0)
    )

    assert isinstance(sample.tdata, np.ndarray)
    assert isinstance(sample.ydata, np.ndarray)
    assert isinstance(sample.params, Params)
    assert isinstance(sample.y0, np.ndarray)

    assert np.array_equal(sample.tdata, [0.0, 1.0, 2.0])
    assert np.array_equal(sample.ydata, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert sample.simulating_func is sim_func
    assert sample.params == Params(k=1.0)
    assert np.array_equal(sample.y0, [0.0, 0.0])


def test_immutability() -> None:
    class Params(NamedTuple):
        k: float

    simu_func = lambda t, y0, k: np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    
    sample = SampleData(
        tdata=[0.0, 1.0, 2.0],
        ydata=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        simulating_func=simu_func,
        params=Params(k=1.0)
    )

    with pytest.raises(AttributeError):
        sample.tdata = None  # type: ignore

    with pytest.raises(AttributeError):
        sample.ydata = None  # type: ignore

    with pytest.raises(AttributeError):
        sample.simulating_func = None  # type: ignore

    with pytest.raises(AttributeError):
        sample.params = None  # type: ignore

    with pytest.raises(AttributeError):
        sample.y0 = None  # type: ignore


def test_recursive_immutability() -> None:
    class Params(NamedTuple):
        k: float

    simu_func = lambda t, y0, k: np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    sample = SampleData(
        tdata=[0.0, 1.0, 2.0],
        ydata=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        simulating_func=simu_func,
        params=Params(k=1.0)
    )

    with pytest.raises(ValueError):
        sample.tdata[0] = 0.0

    with pytest.raises(ValueError):
        sample.ydata[0] = [0.0, 0.0]

    with pytest.raises(TypeError):
        sample.params['k'] = 0.0  # type: ignore

    with pytest.raises(ValueError):
        sample.y0[0] = [0.0, 0.0]


if __name__ == '__main__':
    pytest.main(['-v', __file__])
