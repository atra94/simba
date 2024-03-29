import simba.core as core
from simba.types import float_array, float_base_type


class TFunction(core.SystemComponent):

    def __init__(self, fct, name='TFunction', size=1, dtype=float_base_type):
        out = core.Output(self, name='Out', size=size, dtype=dtype, component_inputs=())
        self._fct = fct
        super().__init__(name, inputs=(), outputs=(out,))

    def compile(self, get_extra_index, numba_compile=True):
        self.output_equation('Out', numba_compile=numba_compile)(self._fct)

    def __call__(self):
        pass

