import numba as nb
import numpy as np

import simba.core as core
from simba.types import float_array


class Clip(core.SystemComponent):

    def __init__(
            self, lower: [float, int, np.ndarray, list, tuple], upper: [float, int, np.ndarray, list, tuple],
            name: str = 'clip', size: int = 1, dtype: type = float_array
    ):

        if isinstance(lower, (float, int)) and isinstance(upper, (float, int)):
            self._array_bounds = False
            self._lower = float(lower)
            self._upper = float(upper)
        else:
            self._array_bounds = True
            self._lower = np.array(lower)
            self._upper = np.array(upper)
            assert len(self._lower.shape) == len(self._upper.shape) == 1, 'Multidimensional bounds are not supported'
            assert len(self._lower) == len(self._upper), 'The length of the bounds is not equal'

        in_ = core.Input(self, name='In', size=size, accepted_dtypes=(dtype,))
        out = core.Output(self, name='Out', size=size, dtype=dtype, system_inputs=(in_,))
        super().__init__(name, inputs=(in_,), outputs=(out,))

    def compile(self, get_extra_index, numba_compile=True):
        lower = self._lower
        upper = self._upper

        def clip(t, in_):
            out = np.empty_like(in_)
            for i in range(len(in_)):
                if in_[i] < lower:
                    out[i] = lower
                elif in_[i] > upper:
                    out[i] = upper
                else:
                    out[i] = in_[i]
            return out

        def clip_array_bounds(t, in_):
            out = np.empty_like(in_)
            for i in range(len(in_)):
                if in_[i] < lower[i]:
                    out[i] = lower
                elif in_[i] > upper[i]:
                    out[i] = upper
                else:
                    out[i] = in_[i]
            return out

        fct = clip_array_bounds if self._array_bounds else clip
        self.output_equation('Out', numba_compile)(fct)

    def __call__(self, in_):
        self._inputs['In'].connect(in_)
