import numba as nb
import numpy as np

from simba.core import SystemComponent, Input, Output
from simba.types import float_array


class PController(SystemComponent):

    @property
    def p_gain(self):
        return self._p_gain

    def __init__(self, name='p_controller', p_gain=1.0):
        self._p_gain = p_gain
        error_input = Input(self, name='error', accepted_dtypes=(float_array,), size=1)
        output = Output(
            self, name='action', dtype=float_array, size=1, signal_names=('action',), system_inputs=(error_input,)
        )
        super().__init__(name, outputs=(output,), inputs=(error_input,))

    def __call__(self, error):
        self._inputs['error'].connect(error)

    def compile(self, numba_compile=True):
        p_gain = np.array(self._p_gain, dtype=np.float64)

        @self.output_equation('action', numba_compile=numba_compile)
        def p_control(t, error_input):
            return p_gain * error_input
