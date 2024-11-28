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
        NaN values are ignored.
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

    Notes
    -----
    `ydata` must have the same shape as the output of `simulating_func`.
    In the case where some species are not measured, the corresponding
    values in `ydata` should be NaN, to keep the shape consistent.
    """
    ysim = simulating_func(tdata, y0, **params_d)

    # Calculate the residuals
    residuals = ysim - ydata  # can include NaN values
    return np.nansum(residuals**2)  # ignore NaN values
