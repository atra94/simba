import numba as nb

import simba.core as core


class TFunction(core.SystemComponent):

    def __init__(self, fct, name='TFunction', size=1, dtype=nb.float64[:]):
        out = core.Output(self, name='Out', size=size, dtype=dtype, system_inputs=())
        self._fct = fct
        super().__init__(name, inputs=(), outputs=(out,))

    def compile(self, numba_compile=True):
        if type(self._fct) is not nb.core.registry.CPUDispatcher and numba_compile:
            try:
                self._fct = nb.njit(self._fct)
            except Exception:
                raise Exception(
                    f'Numba Compilation of the passed function {self._fct.__name__} failed.'
                    f'Try to compile it manually before passing it to the TFunction.'
                )

        self.output_equation('Out', numba_compile=numba_compile)(self._fct)

    def __call__(self):
        pass

