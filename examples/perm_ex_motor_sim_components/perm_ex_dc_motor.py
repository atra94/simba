import numba as nb
import numpy as np

from simba.core import SystemComponent, Input, Output, State
from simba.types import float_base_type


class PermanentlyExcitedDCMotor(SystemComponent):

    @property
    def parameter(self):
        return self._parameter

    _default_motor_parameter = {
        'r_a': 16e-3,
        'l_a': 19e-6,
        'psi_e': 0.165,
        'j_rotor': 0.025
    }

    def __init__(self, name='PermExDCMotor', parameter=None):
        voltage_input = Input(self, name='u',  size=1, dtype=float_base_type)
        speed_input = Input(self, name='omega', size=1, dtype=float_base_type)
        current_output = Output(
            self, name='i', dtype=float_base_type, size=1, signal_names=('i',), component_inputs=()
        )
        torque_output = Output(
            self, name='T', dtype=float_base_type, size=1, signal_names=('T',), component_inputs=()
        )
        state = State(
            self, size=1, component_inputs=(voltage_input, speed_input), dtype=float_base_type
        )
        params = parameter if type(parameter) is dict else dict()
        self._parameter = self._default_motor_parameter.copy()
        self._parameter.update(params)
        super().__init__(
            name, outputs=(current_output, torque_output), inputs=(voltage_input, speed_input), state=state
        )

    def __call__(self, u, omega):
        self._inputs['u'].connect(u)
        self._inputs['omega'].connect(omega)

    def compile(self, get_extra_index, numba_compile=True):

        l_a = self._parameter['l_a']
        r_a = self._parameter['r_a']
        psi_e = self._parameter['psi_e']
        model_parameters = np.array([
            -psi_e, -r_a, 1.0
        ]) / l_a

        @self.state_equation(numba_compile=numba_compile)
        def ode(t, local_state, u, omega):
            # i = local_state[0]
            return model_parameters[0] * omega + model_parameters[1] * local_state + model_parameters[2] * u

        @self.output_equation('T', numba_compile=numba_compile)
        def torque(t, local_state):
            return psi_e * local_state

        @self.output_equation('i', numba_compile=numba_compile)
        def i(t, local_state):
            return local_state
