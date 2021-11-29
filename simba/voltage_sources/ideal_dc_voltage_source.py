import numpy as np

import simba


class IdealDCVoltageSupply(simba.core.SystemComponent):
    """Ideal DC Voltage Supply that supplies with u_nominal independent of the time and the supply current."""

    @property
    def u_nominal(self):
        return self._u_nominal

    @u_nominal.setter
    def u_nominal(self, value):
        self._u_nominal = np.array(value)

    def __init__(self, u_nominal=(600.0,), name='IdealDCVoltageSupply'):
        self._u_nominal = np.array(u_nominal)
        output = simba.core.Output(
            name='u_supply', size=len(self._u_nominal), dtype=simba.types.float_array, component=self,
            system_inputs=()
        )
        super(IdealDCVoltageSupply, self).__init__(name=name, outputs=(output,))

    def compile(self, get_extra_indices, numba_compile=True):
        u_nominal = self._u_nominal

        @self.output_equation('u_supply', numba_compile)
        def u_sup(t):
            return u_nominal
