from collections.abc import Callable, Mapping
from typing import Concatenate

import numpy as np
import numpy.typing as npt


def calc_simulation_rss(
        params_d: Mapping[str, float],
        simulating_func: Callable[
            Concatenate[npt.NDArray, npt.NDArray, ...], npt.NDArray],
        tdata: npt.NDArray, ydata: npt.NDArray, y0: npt.NDArray,
        ) -> float:
    ysim = simulating_func(tdata, y0, **params_d)

    # Calculate the residuals
    residuals = ysim - ydata
    # Remove row with NaN
    residuals = residuals[~np.isnan(residuals).any(axis=1)]
    # Flatten the residuals to 1D array
    residuals = residuals.ravel()

    return np.sum(residuals**2)
