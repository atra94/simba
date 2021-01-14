import numba as nb

from .input import Input


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
    def compiled(self):
        return self._state_function is not None

    def __init__(self, component, inputs, size, signal_names=None, dtype=nb.float64[:]):
        assert all(isinstance(input_, Input) for input_ in inputs)
        self._state_equation = None
        self._signal_names = signal_names
        self._function = None
        self._system_inputs = tuple(inputs)
        self._state_function = None
        self._dtype = dtype
        self._size = size
        self._component = component

    def compile(self):
        assert self.component.local_state_indices is not None, 'State indices have to be set before compilation.'
        assert self.state_equation is not None, 'The state equation has to be set before compilation.'
        state_equation = self._state_equation
        for input_ in self._system_inputs:
            input_.compile()
        input_functions = [input_.input_fct for input_ in self._system_inputs]
        local_state_indices = self.component.local_state_indices

        def state_function(t, global_state):
            local_state = global_state[local_state_indices]
            inputs = tuple(input_fct(t, global_state) for input_fct in input_functions)
            return state_equation(t, local_state, *inputs)

        self._state_function = state_function
