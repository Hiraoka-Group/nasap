from collections.abc import Callable, Iterable, Mapping
from typing import Any, Concatenate

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp


def simulate_with_addition(
        ode_rhs: Callable[
            Concatenate[float, npt.NDArray, ...], npt.NDArray],
        t: npt.NDArray,
        t0: float, y0: npt.NDArray, 
        ode_rhs_args: Iterable | None = None,
        addition: Mapping[Any, npt.NDArray] | None = None,
        *,
        method: str = 'RK45', rtol: float = 1e-3, atol: float = 1e-6,
        ) -> npt.NDArray:
    """Simulate an ODE with addition."""
    _validate_addition(t, t0, y0, addition)
    if ode_rhs_args is None:
        ode_rhs_args = ()
    ode_rhs_args = tuple(ode_rhs_args)
    
    def solve_ivp_wrapper(
            t0: float, y0: npt.NDArray, t_eval: npt.NDArray
            ) -> npt.NDArray:
        sol = solve_ivp(
            ode_rhs, [t0, t_eval[-1]], y0, t_eval=t_eval, 
            args=ode_rhs_args, method=method, rtol=rtol, atol=atol)
        return sol.y.T

    if addition is None or len(addition) == 0:
        return solve_ivp_wrapper(t0, y0, t)

    if t0 in addition:
        raise ValueError(
            f't0 ({t0}) cannot be in the addition times. Use y0 instead.')
    if addition.keys() & set(t):
        raise ValueError(
            f'Addition times {addition.keys()} cannot be in t.'
            f'Use a time point slightly greater than or less than '
            f'the addition times.')

    # Treat y0 as an addition at t0.
    addition = dict(addition)
    addition[t0] = y0
    
    # y_at_cur_t_add contains y just before the next addition.
    # Since y0 will be added at t0, y_at_cur_t_add should be initialized
    # with zeros.
    y_just_before_cur_add = np.zeros_like(y0)

    # Sort the addition times.
    addition_times = sorted(addition.keys())

    y = np.empty((0, *y0.shape))

    for i, t_add in enumerate(addition_times):
        change = addition[t_add]
        
        # Perform the addition.
        y_just_after_cur_add = y_just_before_cur_add + change
        
        # If the last iteration.
        if i == len(addition_times) - 1:
            cur_t_eval = t[t >= t_add]
            cur_y = solve_ivp_wrapper(t_add, y_just_after_cur_add, cur_t_eval)
            y = np.concatenate([y, cur_y])
            break
        
        next_t_add = addition_times[i + 1]
        cur_t_eval = t[(t >= t_add) & (t < next_t_add)]

        # y at next_t_add should be evaluated,
        # which will be y_just_before_cur_add for the next addition.
        cur_t_eval = np.concatenate([cur_t_eval, [next_t_add]])
        
        cur_y = solve_ivp_wrapper(t_add, y_just_after_cur_add, cur_t_eval)
        y_just_before_cur_add = cur_y[-1]  # Update for the next addition.

        # Remove y at next_t_add if it is not originally in t.
        cur_y = cur_y[:-1]
        
        y = np.concatenate([y, cur_y])

    return y


def _validate_addition(
        t: npt.NDArray, t0: float, y0: npt.NDArray,
        addition: Mapping[float, npt.NDArray] | None,
        ):
    if addition is None:
        return
    
    if len(addition) == 0:
        return
    
    for t_add, change in addition.items():
        if t_add < t0:
            raise ValueError(f'time {t_add} is less than t0 ({t0})')
        if t_add > t[-1]:
            raise ValueError(
                f'time {t_add} is greater than t_seq[-1] ({t[-1]})')
        if change.shape != y0.shape:
            raise ValueError(
                f'change shape {change.shape} is different from y0 shape '
                f'{y0.shape}')
