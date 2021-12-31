import numpy as np
from typing import Iterable


class ISliceHolder:

    @property
    def local_slice(self) -> np.ndarray or None:
        return self._local_slice

    @local_slice.setter
    def local_slice(self, value: Iterable[int]):
        self._local_slice = np.asarray(value)
    
    def __init__(self):
        self._local_slice = None

