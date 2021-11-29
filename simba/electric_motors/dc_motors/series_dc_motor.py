import numba as nb
from typing import Callable, Any

import simba
from simba.core import SystemComponent, Input, Output, State
from simba.types import float_array


class SeriesDCMotor(SystemComponent):

    @property
    def parameter(self) -> dict:
        return self._parameter

    _default_motor_parameter = {
        'r_a': 16e-3,
        'r_e': 48e-3,
        'l_a': 19e-6,
        'l_e': 5.4e-3,
        'l_e_prime': 1.7e-3,
        'j_rotor': 0.0025
    }

    def __init__(self, name: str = 'SeriesDCMotor', parameter: dict or None = None):
        voltage_input = Input(self, name='u', accepted_dtypes=(float_array,), size=1)
        speed_input = Input(self, name='omega', accepted_dtypes=(float_array,), size=1)
        current_output = Output(
            self, name='i', dtype=float_array, size=1, signal_names=('i',), system_inputs=()
        )
        torque_output = Output(
            self, name='T', dtype=float_array, size=1, signal_names=('T',), system_inputs=()
        )
        state = State(
            self, size=1, inputs=(voltage_input, speed_input)
        )
        params = parameter if type(parameter) is dict else dict()
        self._parameter = self._default_motor_parameter.copy()
        self._parameter.update(params)
        super().__init__(
            name, outputs=(current_output, torque_output), inputs=(voltage_input, speed_input), state=state
        )

    def __call__(self, u: simba.core.Input, omega: simba.core.Input):
        self._inputs['u'].connect(u)
        self._inputs['omega'].connect(omega)

    def compile(self, get_extra_index: Callable[[Any], int], numba_compile: bool = True):

        l_a = self._parameter['l_a']
        r_a = self._parameter['r_a']
        r_e = self._parameter['r_e']
        l_e = self._parameter['l_e']
        l_e_prime = self._parameter['l_e_prime']
        parameter = nb.float64([
            -l_e_prime, -r_a - r_e, 1.0
        ]) / (l_a + l_e)

        @self.state_equation(numba_compile=numba_compile)
        def ode(t, local_state, u, omega):
            # i = local_state[0]
            return parameter[0] * omega * local_state + parameter[1] * local_state + parameter[2] * u

        @self.output_equation('T', numba_compile=numba_compile)
        def torque(t, local_state):
            return l_e_prime * local_state**2

        @self.output_equation('i', numba_compile=numba_compile)
        def i(t, local_state):
            return local_state
