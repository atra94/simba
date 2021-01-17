import numba as nb

import simba.core as core


class Add(core.SystemComponent):

    def __init__(self, name='add', size=1, dtype=nb.float64[:]):
        in1 = core.Input(self, name='In1', size=size, accepted_dtypes=(dtype,))
        in2 = core.Input(self, name='In2', size=size, accepted_dtypes=(dtype,))
        out = core.Output(self, name='Out', size=size, dtype=dtype, system_inputs=(in1, in2))
        super().__init__(name, inputs=(in1, in2), outputs=(out,))

    def compile(self, numba_compile=True):
        @self.output_equation('Out', numba_compile=numba_compile)
        def subtract(t, in1, in2):
            return in1 + in2

    def __call__(self, in1, in2):
        self._inputs['In1'].connect(in1)
        self._inputs['In2'].connect(in2)

