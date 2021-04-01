import numpy as np

from .input import Input
from .function_factories.state_function_factory import create_state_function
from simba.types import float_array


class State:

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
    def system_inputs(self):
        return self._system_inputs

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
    def local_state_slice(self):
        return self._local_state_slice

    @local_state_slice.setter
    def local_state_slice(self, state_slice):
        self._local_state_slice = np.asarray(state_slice, dtype=np.int32)

    @property
    def compiled(self):
        return self._state_function is not None

    def __init__(self, component, inputs, size, signal_names=None, dtype=float_array):
        assert all(isinstance(input_, Input) for input_ in inputs)
        self._compiled = False
        self._state_equation = None
        self._signal_names = signal_names
        self._function = None
        self._system_inputs = tuple(inputs)
        self._state_function = None
        self._dtype = dtype
        self._size = size
        self._component = component
        self._local_state_slice = None

    def compile(self, global_extra_type):
        if self._compiled:
            return
        assert self._local_state_slice is not None, 'State indices have to be set before compilation.'
        assert self.state_equation is not None, 'The state equation has to be set before compilation.'
        state_equation = self._state_equation
        for input_ in self._system_inputs:
            input_.compile(global_extra_type)
        input_functions = tuple([input_.function for input_ in self._system_inputs])
        self._state_function = create_state_function(
            state_equation, input_functions, self.local_state_slice, global_extra_type, self._component.extra_index
        )
        self._compiled = True
