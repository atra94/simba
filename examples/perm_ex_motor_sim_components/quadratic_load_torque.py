import numba as nb

from simba.core import SystemComponent, Input, Output


class QuadraticLoadTorque(SystemComponent):

    def __init__(self, name='QuadraticLoadTorque', a=0.01, b=0.01, c=0.0, epsilon=1e-4):
        speed_input = Input(self, name='omega', accepted_dtypes=(nb.float64[:],), size=1)
        load_torque_output = Output(
            self, name='T_L', dtype=nb.float64[:], size=1, system_inputs=(speed_input,)
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

    def compile(self, numba_compile=True):
        a = self._a
        b = self._b
        c = self._c
        epsilon = self._epsilon

        @self.output_equation('T_L', numba_compile=numba_compile)
        def load_torque(t, omega):
            om = omega[0]
            sign_omega = 1 if om > epsilon else -1 if om < -epsilon else 0
            return sign_omega * a + omega * b + sign_omega * omega**2 * c
