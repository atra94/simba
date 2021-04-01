import numba as nb
import numpy as np

from simba.core import SystemComponent, Input, Output
from simba.types import float_, float_array


class Logger(SystemComponent):

    def __init__(self, size, dtype, name='Logger'):
        input_ = Input(self, 'In', size, (dtype,))
        output_ = Output(self, 'Out', (input_,), size=0, dtype=nb.none)
        super().__init__(inputs=(input_,), outputs=(output_,), name=name)

    def compile(self, get_extra_index, numba_compile=True):
        t_data = nb.typed.List(lsttype=nb.types.ListType(float_))
        y_data = nb.typed.List(lsttype=nb.types.ListType(self.inputs['In'].dtype))
        self._extra = (t_data, y_data)
        self._extra_index = get_extra_index(self._extra)

        @nb.njit(nb.none(nb.float64))
        def log(t, extra, input_):
            t_data_ = extra[0]
            y_data_ = extra[1]
            t_data_.append(t)
            y_data_.append(input_)

    def reset(self):
        self._extra[0].clear()
        self._extra[1].clear()

    def get_logs(self):
        return {
            't': np.array(self._extra[0]),
            'signals': np.array(self._extra[1])
        }
