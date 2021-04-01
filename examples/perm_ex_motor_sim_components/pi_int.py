import numba as nb
import numpy as np

from simba.core import SystemComponent, Input, Output
from simba.types import float_array, float_


class JitPI(SystemComponent):

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

        self._extra = np.array([
            0.0,  # last t value
            0.0,  # last error
            0.0,  # integrated value
        ])

        p_gain = self._p_gain
        i_gain = self._i_gain

        spec = [
            ('_last_t', float_),
            ('_last_error', float_,),
            ('_integrated_value', float_),
        ]

        @nb.experimental.jitclass(spec)
        class Integrator:
            def __init__(self):
                self._last_t = 0.0
                self._last_error = 0.0
                self._integrated_value = 0.0

            def integrate(self, t, error):
                self._integrated_value = self._integrated_value + (self._last_error + error) / 2 * (t - self._last_t)
                self._last_t = t
                self._last_error = error
                return self._integrated_value

        self._extra = Integrator()
        self._extra_index = get_extra_index(self._extra)
        if numba_compile:
            integrate = nb.njit(float_(nb.typeof(self._extra), float_, float_))(Integrator.integrate)
        else:
            integrate = Integrator.integrate

        @self.output_equation('action', numba_compile=numba_compile)
        def pi_control(t, memory, error_input):
            integrated_value = integrate(memory, t, error_input[0])
            return p_gain * error_input + i_gain * np.array([integrated_value])
