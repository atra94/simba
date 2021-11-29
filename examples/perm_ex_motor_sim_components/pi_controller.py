import numba as nb
import numpy as np

from simba.core import SystemComponent, Input, Output
from simba.types import float_array, float_


class PIController(SystemComponent):

    @property
    def p_gain(self):
        return self._p_gain

    @property
    def i_gain(self):
        return self._i_gain

    def __init__(self, name='pi_controller', p_gain=1.0, i_gain=0.01):
        self._p_gain = p_gain
        self._i_gain = i_gain
        error_input = Input(self, name='error', accepted_dtypes=(float_array,), size=1)
        output = Output(
            self, name='action', dtype=float_array, size=1, signal_names=('action',), system_inputs=(error_input,)
        )
        super().__init__(name, outputs=(output,), inputs=(error_input,))

    def __call__(self, error):
        self._inputs['error'].connect(error)

    def compile(self, get_extra_index, numba_compile=True):

        self._extra = np.ascontiguousarray([
            0.0,  # last t value
            0.0,  # last error
            0.0,  # integrated value
        ])
        self._extra_index = get_extra_index(self._extra)
        self._outputs['action'].extra_index = self._extra_index
        p_gain = self._p_gain
        i_gain = self._i_gain

        def integrate(t, error, memory):
            last_t = memory[0]
            last_error = memory[1]
            integrated_value = memory[2]
            integrated_value += (last_error + error) / 2 * (t - last_t)
            memory[0] = t
            memory[1] = error
            memory[2] = integrated_value
            return integrated_value

        if numba_compile:
            integrate = nb.njit(float_(float_, float_, float_array))(integrate)

        @self.output_equation('action', numba_compile=numba_compile)
        def pi_control(t, memory, error_input):
            integrated_value = integrate(t, error_input[0], memory)
            return p_gain * error_input + i_gain * integrated_value
