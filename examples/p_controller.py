import numba as nb

from simba.core import SystemComponent, Input, Output


class PController(SystemComponent):

    @property
    def p_gain(self):
        return self._p_gain

    def __init__(self, name='p_controller', p_gain=1.0):
        self._p_gain = p_gain
        error_input = Input(self, name='error', dtype=nb.float64[:], size=1)
        output = Output(self, dtype=nb.float64[:], size=1, signal_names=('action',), system_inputs=(error_input,))
        super().__init__(name, outputs=(output,), inputs=(error_input,))

    def compile(self):

        p_gain = self._p_gain

        @nb.njit
        def p_control(local_state, error_input):
            return p_gain * error_input

        self.outputs['action'].equation = p_control

        @nb.njit
        def derivative(local_state, error_input):
            return p_gain

