from simba.core import SystemComponent


class Subsystem(SystemComponent):

    @property
    def components(self):
        return self._components

    def __init__(self, name, components, inputs=(), outputs=()):
        super().__init__(name, inputs, outputs)
        self._components = components
        raise NotImplementedError

    def compile(self, get_extra_indices, numba_compile=True):
        raise NotImplementedError
