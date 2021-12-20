import numpy as np
from typing import Iterable

import simba.core as core


class Subsystem(core.SystemComponent):

    @property
    def components(self):
        return self._components

    def __init__(
            self,
            name: str,
            components: Iterable[core.SystemComponent],
            inputs: Iterable[core.Input] = (),
            outputs: Iterable[core.Output] = ()
    ):
        super().__init__(name, inputs, outputs)
        namelist = [component.name for component in components]
        assert len(set(namelist)) == len(namelist), 'Duplicate names in the components. Use all unique names.'
        self._components = {component.name: component for component in components}

    def compile(self, get_extra_indices, numba_compile=True):
        pass
