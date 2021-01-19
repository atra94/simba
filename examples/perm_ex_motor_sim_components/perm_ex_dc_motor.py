import numba as nb
import numpy as np

from simba.core import SystemComponent, Input, Output, State
from simba.types import float_array


class PermanentlyExcitedDCMotor(SystemComponent):

    @property
    def parameter(self):
        return self._parameter

    _default_motor_parameter = {
        'r_a': 25.0, # Ohm Omega
        'l_a': 3.438e-2,  # Henry H
        'psi_e': 18,  # Weber Wb
        'j_rotor': 0.017,  # kg/ms^2
    }

    def __init__(self, name='PermExDCMotor', parameter=None):
        voltage_input = Input(self, name='u', accepted_dtypes=(float_array,), size=1)
        speed_input = Input(self, name='omega', accepted_dtypes=(float_array,), size=1)
        current_output = Output(
            self, name='i', dtype=nb.float64[:], size=1, signal_names=('i',), system_inputs=()
        )
        torque_output = Output(
            self, name='T', dtype=nb.float64[:], size=1, signal_names=('T',), system_inputs=()
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

    def __call__(self, u, omega):
        self._inputs['u'].connect(u)
        self._inputs['omega'].connect(omega)

    def compile(self, numba_compile=True):

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
