import simba.core as core
from simba.types import float_array


class Gain(core.SystemComponent):

    def __init__(self, gain=1.0, name='Gain', size=1, dtype=float_array):
        in0 = core.Input(self, name='In0', size=size, accepted_dtypes=(dtype,))
        out0 = core.Output(self, name='Out0', size=size, dtype=dtype, system_inputs=(in0,))
        self._gain = gain
        super().__init__(name, inputs=(in0,), outputs=(out0,))

    def compile(self, get_extra_index, numba_compile=True):
        gain = self._gain

        @self.output_equation('Out0', numba_compile=numba_compile)
        def gain_(in0):
            return gain * in0

    def __call__(self, in0):
        self._inputs['In0'].connect(in0)
