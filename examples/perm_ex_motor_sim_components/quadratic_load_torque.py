import numba as nb
import numpy as np

from simba.core import SystemComponent, Input, Output
from simba.types import float_, float_array


class QuadraticLoadTorque(SystemComponent):

    def __init__(self, name='QuadraticLoadTorque', a=0.01, b=0.01, c=0.0, epsilon=1e-4):
        speed_input = Input(self, name='omega', accepted_dtypes=(float_array,), size=1)
        load_torque_output = Output(
            self, name='T_L', dtype=float_array, size=1, system_inputs=(speed_input,)
        )
        self._a = a
        self._b = b
        self._c = c
        self._epsilon = epsilon
        super().__init__(
            name, outputs=(load_torque_output,), inputs=(speed_input,)
        )

    def __call__(self, omega):
        self._inputs['omega'].connect(omega)

    def compile(self, get_extra_index, numba_compile=True):
        a = self._a
        b = self._b
        c = self._c
        epsilon = self._epsilon
        tau = 1e-4
        j_total = 1e-4

        @self.output_equation('T_L', numba_compile=numba_compile)
        def load_torque(t, omega):
            om = omega[0]
            sign_omega = 1 if om > epsilon else -1 if om < -epsilon else 0
            a_ = sign_omega * a \
                if abs(om) > a / j_total * tau \
                else j_total / tau * om
            return float_([a_ + om * b + sign_omega * om**2 * c])
