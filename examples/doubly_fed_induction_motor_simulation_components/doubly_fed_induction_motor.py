import numba as nb
import numpy as np

from simba.core import SystemComponent, Input, Output, State
from simba.types import float_array


class DoublyFedInductionMotor(SystemComponent):

    @property
    def parameter(self):
        return self._parameter

    _default_motor_parameter = {
        'p': 2,
        'l_m': 297.5e-3,
        'l_sig_s': 25.71e-3,
        'l_sig_r': 25.71e-3,
        'j_rotor': 13.695e-3,
        'r_s': 4.42,
        'r_r': 3.51,
    }

    def __init__(self, name='DoublyFedInductionMotor', parameter=None, voltage_space='dq'):
        assert voltage_space in ['dq', 'abc']
        rotor_voltage_input = Input(self, name='u_r', accepted_dtypes=(float_array,), size=2)
        stator_voltage_input = Input(self, name='u_s', accepted_dtypes=(float_array,), size=2)
        speed_input = Input(self, name='omega', accepted_dtypes=(float_array,), size=1)
        stator_dq_current_output = Output(
            self, name='i_r_dq', dtype=nb.float64[:], size=2, signal_names=('i_s_d', 'i_s_q',), system_inputs=()
        )
        stator_abc_current_output = Output(
            self, name='i_r_abc', dtype=nb.float64[:], size=3, signal_names=('i_s_a', 'i_s_b', 'i_s_c'),
            system_inputs=()
        )
        rotor_dq_current_output = Output(
            self, name='i_r_dq', dtype=nb.float64[:], size=2, signal_names=('i_r_d', 'i_r_q',), system_inputs=()
        )
        rotor_abc_current_output = Output(
            self, name='i_r_abc', dtype=nb.float64[:], size=3, signal_names=('i_r_a', 'i_r_b', 'i_r_c'),
            system_inputs=()
        )
        torque_output = Output(
            self, name='T', dtype=nb.float64[:], size=1, signal_names=('T',), system_inputs=()
        )
        epsilon_output = Output(
            self, name='epsilon', dtype=nb.float64[:], size=1, signal_names=('epsilon',), system_inputs=()
        )
        state = State(
            self, size=3, inputs=(rotor_voltage_input, stator_voltage_input, speed_input)
        )
        params = parameter if type(parameter) is dict else dict()
        self._parameter = self._default_motor_parameter.copy()
        self._parameter.update(params)
        super().__init__(
            name,
            outputs=(rotor_abc_current_output, rotor_dq_current_output, torque_output, epsilon_output),
            inputs=(rotor_voltage_input, stator_voltage_input, speed_input),
            state=state
        )

    def __call__(self, u, omega):
        self._inputs['u'].connect(u)
        self._inputs['omega'].connect(omega)

    def compile(self, get_extra_index, numba_compile=True):
        i_s_alpha_idx = 0
        i_s_beta_idx = 1
        psi_r_alpha_idx = 2
        psi_r_beta_idx = 3

        p = self._parameter['p']
        l_m = self._parameter['l_m']
        l_sig_s = self._parameter['l_sig_s']
        l_sig_r = self._parameter['l_sig_r']
        r_s = self._parameter['r_s']
        r_r = self._parameter['r_r']
        l_s = l_m + l_sig_s
        l_r = l_m + l_sig_r
        sigma = (l_s * l_r - l_m ** 2) / (l_s * l_r)
        tau_r = l_r / r_r
        tau_sig = sigma * l_s / (r_s + r_r * (l_m ** 2) / (l_r ** 2))

        model_parameters = np.array([
            [
                0, -1 / tau_sig, 0, l_m * r_r / (sigma * l_s * l_r ** 2), 0, 0, + l_m * p / (sigma * l_r * l_s),
                1 / (sigma * l_s), 0, -l_m / (sigma * l_r * l_s), 0,
            ],  # i_r_alpha_dot
            [
                0, 0, -1 / tau_sig, 0, l_m * r_r / (sigma * l_s * l_r ** 2), l_m * p / (sigma * l_r * l_s), 0, 0,
                1 / (sigma * l_s), 0, -l_m / (sigma * l_r * l_s),
            ],  # i_r_beta_dot
            [
                0, l_m / tau_r, 0, -1 / tau_r, 0, 0, -p, 0, 0, 1, 0,
            ],  # psi_r_alpha_dot
            [
                0, 0, l_m / tau_r, 0, -1 / tau_r, p, 0, 0, 0, 0, 1,],
            # psi_r_beta_dot
            [
                p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],  # epsilon_dot
        ])

        @self.state_equation(numba_compile=numba_compile)
        def ode(t, local_state, u_r_alpha_beta, u_s_alpha_beta, omega):
            omega_ = omega[0]
            variable_vector = np.array([
                omega_,
                local_state[i_s_alpha_idx],  # i_s_alpha
                local_state[i_s_beta_idx],  # i_s_beta
                local_state[psi_r_alpha_idx],  # psi_r_alpha
                local_state[psi_r_beta_idx],  # psi_r_beta
                omega_ * local_state[psi_r_alpha_idx],  # omega * psi_r_alpha
                omega_ * local_state[psi_r_beta_idx],  # omega * psi_r_beta
                u_s_alpha_beta[0],  # u_s_alpha
                u_s_alpha_beta[1],  # u_s_beta
                u_r_alpha_beta[0],  # u_r_alpha
                u_r_alpha_beta[1],  # u_r_beta
            ])
            return np.dot(model_parameters, variable_vector)

        @self.output_equation('T', numba_compile=numba_compile)
        def torque(t, local_state):
            i_s_alpha = local_state[i_s_alpha_idx]
            i_s_beta = local_state[i_s_beta_idx]
            psi_r_alpha = local_state[psi_r_alpha_idx]
            psi_r_beta = local_state[psi_r_beta_idx]
            return np.array([1.5 * p * l_m / (l_m + l_sig_r) * (psi_r_alpha * i_s_beta - psi_r_beta * i_s_alpha)])


