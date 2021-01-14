import numba as nb
import numpy as np

from .output import Output
from .input import Input
from .state import State


class SystemComponent:

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def name(self):
        return self._name

    def __init__(self, name: str, inputs=(), outputs=()):

        assert all(isinstance(o, Output) for o in outputs)
        assert all(isinstance(i, Input) for i in inputs)
        self._outputs = {o.name: o for o in outputs}
        self._inputs = {i.name: i for i in inputs}
        self._name = name
        self._local_state_indices = np.array([], dtype=int)

    def compile(self):
        raise NotImplementedError

    def output_equation(self, func, output_name: str, numba_compile: bool = True):
        if numba_compile:
            func = nb.njit(func)
        self._outputs[output_name] = func


class StatefulSystemComponent(SystemComponent):

    @property
    def state(self):
        return self._state

    @property
    def local_state_indices(self):
        """Returns(np.ndarray(int)): The indices to map the global state vector to the local state of the component."""
        return self._local_state_indices

    @local_state_indices.setter
    def local_state_indices(self, value):
        """Setter for the indices to map the global state vector into the local one of the component. These are set
        by the surrounding system before the compilation of the component.

        Args:
            value(np.ndarray(int)): The index-array.
        """
        self._local_state_indices = np.asarray(value, dtype=np.int_)

    def __init__(self, name: str, state: State, inputs=None, outputs=None):
        super().__init__(name, inputs, outputs)
        assert isinstance(state, State)
        self._state = state

    def state_equation(self, func, numba_compile: bool = True):
        if numba_compile:
            output_dtype = self._state.dtype
            time_dtype = nb.float64
            state_dtype = self._state.dtype
            input_dtypes = tuple(inp.dtype for inp in self._state.system_inputs)
            signature = output_dtype(time_dtype, state_dtype, *input_dtypes)
            func = nb.njit(func, signature)
        self._state.state_equation = func

    def compile(self):
        raise NotImplementedError

