"""Module for calculating the residual sum of squares (RSS) between the
data and the simulation."""

from collections.abc import Callable
from typing import Concatenate, ParamSpec

import numpy as np
import numpy.typing as npt

_P = ParamSpec('_P')

# TODO: Add examples
def calc_simulation_rss(
        tdata: npt.NDArray, ydata: npt.NDArray, 
        simulating_func: Callable[
            Concatenate[npt.NDArray, npt.NDArray, _P], npt.NDArray],
        y0: npt.NDArray, 
        *args: _P.args, **kwargs: _P.kwargs
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

            ``simulating_func(t, y0, *args, **kwargs) -> y``

        where ``t`` is a 1-D array with shape (n,), ``y0`` is a 1-D
        array with shape (m,), ``args`` and ``kwargs`` are additional
        arguments and keyword arguments for the simulation, and ``y``
        is a 2-D array with shape (n, m).
    y0 : npt.NDArray, shape (m,)
        Initial conditions.
    *args
        Additional arguments and keyword arguments for the simulation.
    **kwargs
        Additional arguments and keyword arguments for the simulation.

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
    ysim = simulating_func(tdata, y0, *args, **kwargs)

    # Calculate the residuals
    residuals = ysim - ydata  # can include NaN values
    return np.nansum(residuals**2)  # ignore NaN values
