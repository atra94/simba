from .coordinate_transformation_utils import t_23, q
import numba as nb
import numpy as np
from typing import Callable, Any

import simba
from simba.core import SystemComponent, Input, Output, State
from simba.types import float_, float_array


class PermanentMagnetSynchronousMotor(SystemComponent):

    @property
    def parameter(self) -> dict:
        return self._parameter

    @property
    def input_space(self) -> str:
        return self._input_space

    _default_motor_parameter = {
        'p': 3,
        'l_d': 0.37e-3,
        'l_q': 1.2e-3,
        'j_rotor': 0.03883,
        'r_s': 18e-3,
        'psi_p': 66e-3,
    }

    def __init__(
            self, name: str = 'PermanentMagnetSynchronousMotor', parameter: dict or None = None, input_space='abc'
        ):
        assert input_space in ['dq', 'abc', 'alphabeta']
        self._input_space = input_space
        if input_space in ['dq', 'alphabeta']:
            voltage_input = Input(self, name='u', accepted_dtypes=(float_array,), size=2)
        else:
            voltage_input = Input(self, name='u', accepted_dtypes=(float_array,), size=3)
        speed_input = Input(self, name='omega', accepted_dtypes=(float_array,), size=1)
        current_output = Output(
            self, name='i_dq', dtype=float_array, size=2, signal_names=('i_sd', 'i_sq'), system_inputs=()
        )
        direct_current_output = Output(
            self, name='i_sd', dtype=float_array, size=1, signal_names=('i_sd',), system_inputs=()
        )
        quadrature_current_output = Output(
            self, name='i_sq', dtype=float_array, size=1, signal_names=('i_sq',), system_inputs=()
        )
        abc_current_output = Output(
            self, name='i_abc', dtype=float_array, size=3, signal_names=('i_a', 'i_b', 'i_c'), system_inputs=()
        )
        torque_output = Output(
            self, name='T', dtype=float_array, size=1, signal_names=('T',), system_inputs=()
        )
        epsilon_output = Output(
            self, name='epsilon_el', dtype=float_array, size=1, signal_names=('epsilon_el', ),
            system_inputs=()
        )
        cos_sin_epsilon_output = Output(
            self, name='cos_sin_epsilon_el', dtype=float_array, size=2,
            signal_names=('cos_epsilon_el', 'sin_epsilon_el'), system_inputs=()
        )

        state = State(
            self, size=3, inputs=(voltage_input, speed_input)
        )
        params = parameter if type(parameter) is dict else dict()
        self._parameter = self._default_motor_parameter.copy()
        self._parameter.update(params)
        super().__init__(
            name, outputs=(
                current_output, direct_current_output, quadrature_current_output, torque_output, abc_current_output,
                cos_sin_epsilon_output, epsilon_output
            ),
            inputs=(voltage_input, speed_input), state=state
        )

    def __call__(self, u: simba.core.Input, omega: simba.core.Input):
        self._inputs['u'].connect(u)
        self._inputs['omega'].connect(omega)

    def compile(self, get_extra_index: Callable[[Any], int], numba_compile: bool = True):
        i_sd_state_idx = 0
        i_sq_state_idx = 1
        epsilon_el_state_idx = 2
        u_sd_input_idx = 0
        u_sq_input_idx = 1

        l_d = self._parameter['l_d']
        l_q = self._parameter['l_q']
        r_s = self._parameter['r_s']

        psi_p = self._parameter['psi_p']
        p = self._parameter['p']

        t_23_ = nb.njit(float_array)(t_23) if numba_compile else t_23
        q_ = nb.njit(q) if numba_compile else q
        signature = float_array(float_, float_array, float_array, float_array)

        def ode(t, local_state, u, omega):
            return np.dot(
                model_parameter,
                nb.float64([
                    local_state[i_sd_state_idx],
                    local_state[i_sq_state_idx],
                    omega * local_state[i_e_state_idx],
                    u[u_a_input_idx],
                    u[u_e_input_idx],
                ])
            )
        ode_ = nb.njit(signature)(ode) if numba_compile else ode

        def alphabeta_to_dq(t, local_state, u, omega):
            epsilon = local_state[-1]
            return ode_(t, local_state, q_(u, epsilon), omega)

        alphabeta_to_dq_ = nb.njit(signature)(alphabeta_to_dq) if numba_compile else alphabeta_to_dq

        def abc_to_alphabeta(t, local_state, u, omega):
            return alphabeta_to_dq(t, local_state, t_23_(u), omega)

        abc_to_alphabeta_ = nb.njit(signature)(alphabeta_to_dq_) if numba_compile else alphabeta_to_dq_

        state_equation = ode_ if self._input_space == 'dq' \
            else alphabeta_to_dq_ if self._input_space == 'alphabeta' \
            else abc_to_alphabeta_

        # Already manually compiled, if necessary
        self.state_equation(False)(state_equation)

        @self.output_equation('T', numba_compile=numba_compile)
        def torque(t, local_state):
            return l_e_prime * local_state[i_a_state_idx] * local_state[i_e_state_idx]

        @self.output_equation('i', numba_compile=numba_compile)
        def i(t, local_state):
            return local_state

        @self.output_equation('i_a', numba_compile=numba_compile)
        def i_a(t, local_state):
            return local_state[[i_a_state_idx]]

        @self.output_equation('i_e', numba_compile=numba_compile)
        def i_e(t, local_state):
            return local_state[[i_e_state_idx]]
