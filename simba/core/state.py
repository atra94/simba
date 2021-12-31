import numpy as np

from .input import Input
from .function_factories.state_function_factory import create_state_function
from simba.types import float_base_type
from simba.core.interfaces import ISliceHolder
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from simba.core import Input

class State(ISliceHolder):

    @property
    def component(self):
        return self._component

    @property
    def size(self):
        return self._size

    @property
    def dtype(self):
        return self._dtype

    @property
    def signal_names(self):
        return self._signal_names

    @property
    def function(self):
        return self._function

    @property
    def component_inputs(self) -> Iterable['Input']:
        return self._component_inputs

    @property
    def state_equation(self):
        return self._state_equation

    @state_equation.setter
    def state_equation(self, equation):
        assert not self.compiled, 'Cannot change the state equation after the compilation.'
        self._state_equation = equation

    @property
    def state_function(self):
        return self._state_function

    @property
    def compiled(self):
        return self._state_function is not None

    def __init__(self, component, component_inputs, size, signal_names=None, dtype=float_base_type):
        ISliceHolder.__init__(self)
        assert all(isinstance(input_, Input) for input_ in component_inputs)
        self._compiled = False
        self._state_equation = None
        self._signal_names = signal_names
        self._function = None
        self._component_inputs = tuple(component_inputs)
        self._state_function = None
        self._dtype = dtype
        self._size = size
        self._component = component

    def compile(self, global_extra_type):
        if self._compiled:
            return
        assert self._local_slice is not None, 'State indices have to be set before compilation.'
        assert self.state_equation is not None, 'The state equation has to be set before compilation.'
        self._state_function = create_state_function(self, global_extra_type)
        self._compiled = True
