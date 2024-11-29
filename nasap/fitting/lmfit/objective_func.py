from collections.abc import Callable
from typing import Concatenate

import numpy.typing as npt
from lmfit import Parameters

from ..rss import calc_simulation_rss


def make_objective_func_for_lmfit_minimizer(
        tdata: npt.NDArray, ydata: npt.NDArray, 
        simulating_func: Callable[
            Concatenate[npt.NDArray, npt.NDArray, ...], npt.NDArray],
        y0: npt.NDArray,
        ) -> Callable[[npt.NDArray], float]:
    """Make an objective function for the `lmfit.Minimizer` class from 
    a simulating function.

    Resulting objective function can be used as the ``userfcn`` parameter
    of the `lmfit.Minimizer` class.
    
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
        of the simulation will be passed as ``*kwargs``, i.e., 
        positional-only arguments are not supported.
    y0 : npt.NDArray, shape (m,)
        Initial conditions.

    Returns
    -------
    Callable
        The objective function that calculates the residual sum of squares
        (RSS) between the data and the simulation for a given set of 
        parameters. It has the signature
        
            ``objective_func(params) -> rss``

        where ``params`` is a `lmfit.Parameters` object and ``rss`` is
        the residual sum of squares (float). The parameter object should
        contain all the parameters that are expected by the simulation
        function.
    
    Examples
    --------
    Import required modules and functions:
    >>> import numpy as np
    >>> from lmfit import Parameters
    >>> from lmfit import Minimizer
    >>> from nasap.fitting.lmfit import make_objective_func_for_lmfit_minimizer
    >>> from nasap.simulation import make_simulating_func_from_ode_rhs
    
    Define the ODE right-hand side function:
    >>> def ode_rhs(t, y, k):
    ...     return np.array([-k * y[0], k * y[0]])

    Make a simulating function from the ODE right-hand side function:
    >>> simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    Define the data:
    >>> tdata = np.array([0, 0.1, 0.2])
    >>> ydata = np.array([[1, 0], [0.9, 0.1], [0.82, 0.18]])

    Define the initial conditions:
    >>> y0 = np.array([1, 0])

    Make the objective function:
    >>> objective_func = make_objective_func_for_lmfit_minimizer(
    ...     tdata, ydata, simulating_func, y0)

    Define the parameters:
    >>> params = Parameters()
    >>> params.add('k', value=0.1)

    Make the minimizer:
    >>> minimizer = Minimizer(objective_func, params)

    Run the minimizer:
    >>> result = minimizer.minimize()
    >>> result.params['k'].value
    np.float64(1.006577440063097)  # may vary slightly
    """
    def objective_func(params: Parameters) -> float:
        return calc_simulation_rss(
            tdata, ydata, simulating_func, y0, **params.valuesdict())
    
    return objective_func
