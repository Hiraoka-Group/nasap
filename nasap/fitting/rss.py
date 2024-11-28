"""Module for calculating the residual sum of squares (RSS) between the
data and the simulation."""

from collections.abc import Callable, Mapping
from typing import Concatenate

import numpy as np
import numpy.typing as npt


def calc_simulation_rss(
        tdata: npt.NDArray, ydata: npt.NDArray, 
        simulating_func: Callable[
            Concatenate[npt.NDArray, npt.NDArray, ...], npt.NDArray],
        y0: npt.NDArray, params_d: Mapping[str, float],
        ) -> float:
    """Calculate the residual sum of squares (RSS) between the data 
    and the simulation.

    Parameters
    ----------
    tdata : npt.NDArray, shape (n,)
        Time points of the data.
    ydata : npt.NDArray, shape (n, m)
        Data to be compared with the simulation.
        If there are NaN values in the data, the corresponding rows
        will be removed.
    simulating_func : Callable
        Function that simulates the system. 

            ``simulating_func(t, y0, **params_d) -> y``

        where ``t`` is a 1-D array with shape (n,), ``y0`` is a 1-D
        array with shape (m,), ``params_d`` is a dictionary with the
        parameters of the system, and ``y`` is a 2-D array with shape
        (n, m).
    y0 : npt.NDArray, shape (m,)
        Initial conditions.
    params_d : Mapping[str, float]
        Parameters of the system.

    Returns
    -------
    float
        The residual sum of squares (RSS) between the data and the 
        simulation.
    """
    ysim = simulating_func(tdata, y0, **params_d)

    # Calculate the residuals
    residuals = ysim - ydata
    # Remove row with NaN
    residuals = residuals[~np.isnan(residuals).any(axis=1)]
    # Flatten the residuals to 1D array
    residuals = residuals.ravel()

    return np.sum(residuals**2)
