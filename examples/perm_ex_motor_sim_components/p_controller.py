import numba as nb
import numpy as np

from simba.core import SystemComponent, Input, Output
from simba.types import float_base_type


class PController(SystemComponent):

    @property
    def p_gain(self):
        return self._p_gain

    def __init__(self, name='p_controller', p_gain=1.0):
        self._p_gain = p_gain
        error_input = Input(self, name='error', dtype=float_base_type, size=1)
        output = Output(
            self, name='action', dtype=float_base_type, size=1, signal_names=('action',),
            component_inputs=(error_input,)
        )
        super().__init__(name, outputs=(output,), inputs=(error_input,))

    def __call__(self, error):
        self._inputs['error'].connect(error)

    def compile(self, get_extra_index, numba_compile=True):
        p_gain = self._p_gain

        @self.output_equation('action', numba_compile=numba_compile)
        def p_control(t, error_input):
            return p_gain * error_input
