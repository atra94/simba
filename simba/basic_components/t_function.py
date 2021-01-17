import numba as nb

import simba.core as core


class TFunction(core.SystemComponent):

    def __init__(self, fct, name='TFunction', size=1, dtype=nb.types.Array(nb.float32, 1, 'C')):
        out = core.Output(self, name='Out', size=size, dtype=dtype, system_inputs=())
        self._fct = fct
        super().__init__(name, inputs=(), outputs=(out,))

    def compile(self, numba_compile=True):
        self.output_equation('Out', numba_compile=numba_compile)(self._fct)

    def __call__(self):
        pass

