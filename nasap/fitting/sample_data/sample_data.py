from typing import Generic, NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt

from nasap.simulation import SimulatingFunc

_T = TypeVar('_T', bound=SimulatingFunc)
_S = TypeVar('_S', bound=NamedTuple)


class SampleData(Generic[_T, _S]):
    """Immutable class for sample data."""
    def __init__(
            self, tdata: npt.ArrayLike, ydata: npt.ArrayLike,
            simulating_func: _T,
            params: _S) -> None:
        
        self._tdata = np.array(tdata)
        self._ydata = np.array(ydata)
        self._simulating_func = simulating_func

        self._params = params
    
        self._tdata.flags.writeable = False
        self._ydata.flags.writeable = False

    @property
    def tdata(self) -> npt.NDArray:
        """Time points of the data. (Read-only)"""
        return self._tdata
    
    @property
    def ydata(self) -> npt.NDArray:
        """Data to be compared with the simulation. (Read-only)"""
        return self._ydata
    
    @property
    def simulating_func(self) -> _T:
        """Function that simulates the system."""
        return self._simulating_func
    
    @property
    def params(self) -> _S:
        """NamedTuple of parameters. (Read-only)"""
        return self._params

    @property
    def y0(self) -> npt.NDArray:
        """Initial conditions. (Read-only)"""
        return self.ydata[0]
