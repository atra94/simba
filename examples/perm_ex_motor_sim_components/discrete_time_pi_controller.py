import numba as nb
import numpy as np

from simba.core import SystemComponent, Input, Output
from simba.types import float_array, float_base_type
from simba.config import get_t_diff


class DiscreteTimePIController(SystemComponent):

    @property
    def p_gain(self):
        return self._p_gain

    @property
    def i_gain(self):
        return self._i_gain

    @property
    def tau(self):
        return self._tau

    def __init__(self, name='pi_controller', p_gain=1.0, i_gain=0.01, tau=1e-4):
        self._p_gain = p_gain
        self._i_gain = i_gain
        self._tau = tau
        error_input = Input(self, name='error', dtype=float_base_type, size=1)
        output = Output(
            self, name='action', dtype=float_base_type, size=1, signal_names=('action',),
            component_inputs=(error_input,)
        )
        super().__init__(name, outputs=(output,), inputs=(error_input,))

    def __call__(self, error):
        self._inputs['error'].connect(error)

    def compile(self, get_extra_index_callback, numba_compile=True):
        self._extra = np.array([
            0.0,  # last t value
            0.0,  # integrated value
        ])
        self._extra_index = get_extra_index_callback(self._extra)
        p_gain = self._p_gain
        i_gain = self._i_gain
        tau = self._tau
        time_jitter = get_t_diff()

        assert time_jitter < tau, 'The discrete time constant tau has to be greater than the global time jitter.'

        def integrate(t, error, memory):
            integrated_value = memory[1]
            integrated_value += error * tau
            memory[0] = t
            memory[1] = integrated_value
            return integrated_value

        if numba_compile:
            integrate = nb.njit(float_base_type(float_base_type, float_base_type, float_array))(integrate)

        @self.output_equation('action', numba_compile=numba_compile)
        def pi_control(t, memory, error_input):
            last_t = memory[0]
            if last_t + tau < t - time_jitter:
                integrate(t, error_input[0], memory)
            integrated_value = memory[1]
            return p_gain * error_input + i_gain * integrated_value
