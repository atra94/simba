import numba as nb
import numpy as np

from .output import Output
from .input import Input
from .state import State
from simba.types import float_base_type, float_array


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

    @property
    def state(self):
        return self._state

    @property
    def local_state_slice(self):
        return self._state.local_slice if self._state is not None else np.arange(0)

    @property
    def extra_index(self):
        return self._extra_index

    @property
    def extra(self):
        return self._extra

    def __init__(self, name: str, inputs=(), outputs=(), state=None):
        assert all(isinstance(o, Output) for o in outputs)
        assert all(isinstance(i, Input) for i in inputs)
        assert state is None or isinstance(state, State)
        self._outputs = {o.name: o for o in outputs}
        self._inputs = {i.name: i for i in inputs}
        self._name = name
        self._local_state_indices = np.array([], dtype=int)
        self._state = state
        self._extra_index = None
        self._extra = None

    def compile(self, get_extra_indices, numba_compile=True):
        raise NotImplementedError

    def output_equation(self, output_name: str, numba_compile: bool = True):

        def wrapper(func):
            if numba_compile:
                output_dtype = self._outputs[output_name].dtype[:]
                time_dtype = float_base_type
                input_signature = [time_dtype]
                if self._state is not None:
                    input_signature.append(self._state.dtype[:])
                if self._extra is not None:
                    input_signature.append(nb.typeof(self._extra))
                for inp in self._outputs[output_name].component_inputs:
                    if inp.connected:
                        input_signature.append(inp.dtype[:])
                    else:
                        input_signature.append(inp.dtype[::1])
                input_signature = tuple(input_signature)
                signature = output_dtype(*input_signature)
                func = nb.njit(signature)(func)
            self._outputs[output_name].output_equation = func

        return wrapper

    def state_equation(self, numba_compile: bool = True):
        assert self._state is not None

        def wrapper(func):
            if numba_compile:
                time_dtype = float_base_type
                input_signature = [time_dtype, self._state.dtype[:]]
                if self._extra is not None:
                    input_signature.append(nb.typeof(self._extra))
                for inp in self._state.component_inputs:
                    input_signature.append(inp.dtype[:])
                input_signature = tuple(input_signature)
                signature = float_array(*input_signature)
                func = nb.njit(signature)(func)
            self._state.state_equation = func

        return wrapper
