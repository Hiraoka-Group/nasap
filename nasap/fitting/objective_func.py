from collections.abc import Callable
from typing import Concatenate

import numpy.typing as npt

from .rss import calc_simulation_rss


# TODO: Add examples
def make_objective_func_from_simulating_func(
        tdata: npt.NDArray, ydata: npt.NDArray, 
        simulating_func: Callable[
            Concatenate[npt.NDArray, npt.NDArray, ...], npt.NDArray],
        y0: npt.NDArray,
        ) -> Callable[[npt.NDArray], float]:
    """Make an objective function from a simulating function.
    
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
        is a 2-D array with shape (n, m). Note that all the parameters
        of the simulation will be passed as ``*args``, i.e., keyword-only
        arguments are not supported.
    y0 : npt.NDArray, shape (m,)
        Initial conditions.

    Returns
    -------
    Callable
        The objective function that calculates the residual sum of squares
        (RSS) between the data and the simulation for a given set of parameters.
        It has the signature
        
            ``objective_func(param_values) -> rss``
        
        where ``param_values`` is a 1-D array with shape (p,) and ``rss`` is
        the residual sum of squares (float). The parameters should be passed
        in the same order as they are expected by the simulation function.
    """

    def objective_func(param_values: npt.NDArray) -> float:
        rss = calc_simulation_rss(tdata, ydata, simulating_func, y0, *param_values)
        return rss
    
    return objective_func
