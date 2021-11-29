import numpy as np

import simba


class InnerResistanceDCVoltageSupply(simba.core.SystemComponent):

    @property
    def u_nominal(self):
        return self._u_nominal

    @u_nominal.setter
    def u_nominal(self, value):
        self._u_nominal = np.array(value)

    @property
    def r_inner(self):
        return self._r_inner

    @r_inner.setter
    def r_inner(self, value):
        self._r_inner = np.array(value)

    def __init__(self, u_nominal=(600.0,), r_inner=(0.1,), name='IdealDCVoltageSupply'):
        self._u_nominal = np.array(u_nominal)
        self._r_inner = np.array(r_inner)
        assert self.r_inner.shape == self.u_nominal.shape, 'Inner resistance and nominal voltage need the same shape.'
        current_input_ = simba.core.Input(
            component=self, name='i_supply', size=len(r_inner), default_value=np.zeros_like(u_nominal),
            accepted_dtypes=(simba.types.float_, simba.types.float_array)
        )
        voltage_output = simba.core.Output(
            name='u_supply', size=len(self._u_nominal), dtype=simba.types.float_array, component=self,
            system_inputs=(current_input_,)
        )
        super().__init__(name=name, outputs=(voltage_output,), inputs=(current_input_,))

    def compile(self, get_extra_indices, numba_compile=True):
        u_nominal = self._u_nominal
        r_inner = self._r_inner

        @self.output_equation('u_supply', numba_compile)
        def u_sup(t, i_supply):
            return u_nominal - r_inner * i_supply
