import numba as nb
import numpy as np

from simba.core import SystemComponent, Input, Output, State
from simba.types import float_base_type


class RotationalMechanicalLoad(SystemComponent):

    def __init__(self, name='RotationalLoad', j=0.05):
        driving_torque_input = Input(self, name='T', size=1, dtype=float_base_type)
        load_torque_input = Input(
            self, name='T_L', size=1, default_value=np.array([0.0]), dtype=float_base_type
        )
        speed_output = Output(
            self, name='omega', dtype=float_base_type, size=1, signal_names=('omega',), component_inputs=()
        )
        state = State(
            self, size=1, component_inputs=(driving_torque_input, load_torque_input)
        )
        self._j = j

        super().__init__(
            name, outputs=(speed_output,), inputs=(driving_torque_input, load_torque_input,), state=state
        )

    def __call__(self, t, t_l):
        self._inputs['T'].connect(t)
        self._inputs['T_L'].connect(t_l)

    def compile(self, get_extra_index, numba_compile=True):
        j = self._j

        @self.state_equation(numba_compile=numba_compile)
        def ode(t, state, driving_torque, load_torque):
            return (driving_torque - load_torque) / j

        @self.output_equation('omega', numba_compile=numba_compile)
        def omega(t, local_state):
            return local_state
