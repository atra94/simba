import numba as nb

from .input import Input


class State:

    @property
    def dtype(self):
        return self._dtype

    @property
    def state_equation(self):
        return self._state_equation

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
    def global_state_indices(self):
        return self._global_state_indices

    @global_state_indices.setter
    def global_state_indices(self, value):
        self._global_state_indices = np.asarray(value, dtype=np.int32)

    @property
    def size(self):
        return self._size

    def __init__(self, state_equation, inputs, size, signal_names=None, dtype=nb.float64[:]):
        self._state_equation = state_equation
        self._signal_names = signal_names
        self._function = None
        self._system_inputs = tuple(inputs)
        self._state_function = None
        self._dtype = dtype
        self._size = size
        self._global_state_indices = None
        assert all(isinstance(input_, Input) for input_ in self._system_inputs)

    def compile(self):
        assert self.global_state_indices is not None, 'State indices have to be set before compilation.'

        state_equation = self._state_equation
        input_functions = [input_.input_fct for input_ in self._system_inputs]
        global_state_indices = self._global_state_indices

        def state_function(global_state):
            local_state = global_state[global_state_indices]
            inputs = tuple(input_fct(global_state) for input_fct in input_functions)
            return state_equation(local_state, *inputs)

        self._state_function = state_function
