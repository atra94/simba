import simba.core as core
from simba.types import float_array, float_base_type


class Sub(core.SystemComponent):

    def __init__(self, name='sub', size=1, dtype=float_base_type):
        in1 = core.Input(self, name='In1', size=size, dtype=dtype)
        in2 = core.Input(self, name='In2', size=size, dtype=dtype)
        out = core.Output(self, name='Out', size=size, dtype=dtype, component_inputs=(in1, in2))
        super().__init__(name, inputs=(in1, in2), outputs=(out,))

    def compile(self, get_extra_index, numba_compile=True):
        @self.output_equation('Out', numba_compile=numba_compile)
        def subtract(t, in1, in2):
            return in1 - in2

    def __call__(self, in1, in2):
        self._inputs['In1'].connect(in1)
        self._inputs['In2'].connect(in2)
