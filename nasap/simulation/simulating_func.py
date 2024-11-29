"""
Module for making simulating functions from ODE right-hand side functions.
"""

from collections.abc import Callable
from typing import Concatenate, ParamSpec

import numpy.typing as npt
from scipy.integrate import solve_ivp

_P = ParamSpec('_P')

def make_simulating_func_from_ode_rhs(
        ode_rhs: Callable[Concatenate[float, npt.NDArray, _P], npt.NDArray]
        ) -> Callable[Concatenate[npt.NDArray, npt.NDArray, _P], npt.NDArray]:
    """Make a simulating function from an ODE right-hand side function.

    Resulting simulating function can be called with time points, 
    initial values of the dependent variables, and the parameters of 
    the ODE right-hand side function. For example, it can be called
    as ``simulating_func(t, y0, k1, k2, k3, k4)`` for a 4-parameter 
    ODE right-hand side function ``ode_rhs(t, y, k1, k2, k3, k4)``.

    Parameters
    ----------
    ode_rhs: Callable
        The ODE right-hand side function. It should have the signature
        
            ``ode_rhs(t, y, *args, **kwargs) -> dydt``

        where ``t`` is the time (float), ``y`` is the dependent
        variable (1-D array, shape (n,)), ``args`` and ``kwargs``
        are the parameters of the ODE right-hand side function, and
        ``dydt`` is the derivative of the dependent variable (1-D
        array, shape (n,)). The names of ``t`` and ``y`` can be
        different.

    Returns
    -------
    Callable
        The simulating function for the ODE right-hand side.
        It has the signature
        
            ``simulating_func(t, y0, *args, **kwargs) -> y``
    
        where ``t`` is the time points (1-D array, shape (m,)),
        ``y0`` is the initial values of the dependent variables
        (1-D array, shape (n,)), ``args`` and ``kwargs`` are the
        parameters of the ODE right-hand side function, and ``y``
        is the dependent variables at the time points (2-D array,
        shape (m, n)).
    """
    def simulating_func(
            t: npt.NDArray, y0: npt.NDArray,
            *args: _P.args, **kwargs: _P.kwargs
            ) -> npt.NDArray:
        def ode_rhs_with_fixed_parameters(
                t: float, y: npt.NDArray) -> npt.NDArray:
            return ode_rhs(t, y, *args, **kwargs)

        sol = solve_ivp(
            ode_rhs_with_fixed_parameters, (t[0], t[-1]), y0,
            dense_output=True)

        return sol.sol(t)

    return simulating_func
