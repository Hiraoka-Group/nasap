from types import MappingProxyType
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from frozendict import frozendict

_T = TypeVar('_T')


class SampleData(Generic[_T]):
    """Immutable class for sample data."""
    def __init__(
            self, tdata: npt.ArrayLike, ydata: npt.ArrayLike,
            simulating_func: _T,
            params: dict[str, float] | None = None) -> None:
        self._tdata = np.array(tdata)
        self._ydata = np.array(ydata)
        self._simulating_func = simulating_func

        if params is None:
            params = {}
        self._params = frozendict(params)
    
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
    def params(self) -> MappingProxyType[str, float]:
        """Read-only dictionary of parameters."""
        return MappingProxyType(self._params)

    @property
    def y0(self) -> npt.NDArray:
        """Initial conditions. (Read-only)"""
        return self.ydata[0]
