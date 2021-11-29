import numba as nb
import numpy as np
from typing import Callable, Any

import simba
from simba.core import SystemComponent, Input, Output, State
from simba.types import float_, float_array


class ShuntDCMotor(SystemComponent):

    @property
    def parameter(self) -> dict:
        return self._parameter

    _default_motor_parameter = {
        'r_a': 16e-3,
        'r_e': 4e-1,
        'l_a': 19e-6,
        'l_e': 5.4e-3,
        'l_e_prime': 1.7e-3,
        'j_rotor': 0.0025
    }

    def __init__(self, name: str = 'ShuntDCMotor', parameter: dict or None = None):
        voltage_input = Input(self, name='u', accepted_dtypes=(float_array,), size=1)
        speed_input = Input(self, name='omega', accepted_dtypes=(float_array,), size=1)
        current_output = Output(
            self, name='i', dtype=float_array, size=2, signal_names=('i_a', 'i_e'), system_inputs=()
        )
        anchor_current_output = Output(
            self, name='i_a', dtype=float_array, size=1, signal_names=('i_a',), system_inputs=()
        )
        exciting_current_output = Output(
            self, name='i_e', dtype=float_array, size=1, signal_names=('i_e',), system_inputs=()
        )
        sum_current_output = Output(
            self, name='i_sum', dtype=float_array, size=1, signal_names=('i_sum',), system_inputs=()
        )
        torque_output = Output(
            self, name='T', dtype=float_array, size=1, signal_names=('T',), system_inputs=()
        )
        state = State(
            self, size=2, inputs=(voltage_input, speed_input)
        )
        params = parameter if type(parameter) is dict else dict()
        self._parameter = self._default_motor_parameter.copy()
        self._parameter.update(params)
        super().__init__(
            name, outputs=(
                current_output, anchor_current_output, exciting_current_output, sum_current_output, torque_output
            ),
            inputs=(voltage_input, speed_input), state=state
        )

    def __call__(self, u: simba.core.Input, omega: simba.core.Input):
        self._inputs['u'].connect(u)
        self._inputs['omega'].connect(omega)

    def compile(self, get_extra_index: Callable[[Any], int], numba_compile: bool = True):
        i_a_state_idx = 0
        i_e_state_idx = 1
        l_a = self._parameter['l_a']
        r_a = self._parameter['r_a']
        r_e = self._parameter['r_e']
        l_e = self._parameter['l_e']
        l_e_prime = self._parameter['l_e_prime']
        model_parameter = nb.float64(
            [
                [-r_a, 0, -l_e_prime, 1],
                [0, -r_e, 0, 1]
            ]
        )
        model_parameter[i_a_state_idx] = model_parameter[i_a_state_idx] / l_a
        model_parameter[i_e_state_idx] = model_parameter[i_e_state_idx] / l_e

        @self.state_equation(numba_compile=numba_compile)
        def ode(t, local_state, u, omega):
            return float_([np.dot(
                model_parameter,
                np.concatenate([
                    local_state,
                    omega * local_state[i_e_state_idx],
                    u
                ])
            )])

        @self.output_equation('T', numba_compile=numba_compile)
        def torque(t, local_state):
            return l_e_prime * local_state[i_a_state_idx] * local_state[i_e_state_idx]

        @self.output_equation('i', numba_compile=numba_compile)
        def i(t, local_state):
            return local_state

        @self.output_equation('i_sum', numba_compile=numba_compile)
        def i_sum(t, local_state):
            return nb.float64([local_state[i_a_state_idx] + local_state[i_e_state_idx]])

        @self.output_equation('i_a', numba_compile=numba_compile)
        def i_a(t, local_state):
            return local_state[[i_a_state_idx]]

        @self.output_equation('i_e', numba_compile=numba_compile)
        def i_e(t, local_state):
            return local_state[[i_e_state_idx]]
