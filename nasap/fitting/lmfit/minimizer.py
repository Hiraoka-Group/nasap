import inspect
from collections.abc import Callable, Iterable
from typing import Concatenate, ParamSpec, Protocol, TypeAlias, TypeVar

import numpy.typing as npt
from lmfit import Minimizer, Parameters

from ..objective_func import make_objective_func_from_simulating_func
from ..rss import calc_simulation_rss

_P = ParamSpec('_P')


def make_lmfit_minimizer(
        tdata: npt.NDArray, ydata: npt.NDArray,
        simulating_func: Callable[
            Concatenate[npt.NDArray, npt.NDArray, _P], npt.NDArray],
        y0: npt.NDArray,
        params: Parameters,
        ) -> Minimizer:
    """Make a `lmfit.Minimizer` object from a simulating function.

    Resulting minimizer can be used to minimize the residual sum of 
    squares (RSS) between the data and the simulation.

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
        
        where ``t`` is the time points (1-D array, shape (n,)), 
        ``y0`` is the initial values of the dependent variables
        (1-D array, shape (m,)), ``args`` and ``kwargs`` are the
        parameters of the simulation, and ``y`` is the dependent
        variables at the time points (2-D array, shape (n, m)).
    y0 : npt.NDArray, shape (m,)
        Initial conditions.
    params : lmfit.Parameters
        Parameters for the minimization. The names of the parameters
        should match the names of the arguments of the simulation
        function.

    Returns
    -------
    lmfit.Minimizer
        The minimizer object that can be used to minimize the residual
        sum of squares (RSS) between the data and the simulation.

    Examples
    --------
    """
    
    def func_for_minimizer(params: Parameters) -> float:
        return calc_simulation_rss(
            tdata, ydata, simulating_func, y0, **params.valuesdict())

    return Minimizer(func_for_minimizer, params)
