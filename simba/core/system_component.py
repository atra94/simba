import numba as nb
import numpy as np

from .output import Output
from .input import Input
from .state import State
from simba.types import float_


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
        return self._state.local_state_slice if self._state is not None else np.arange(0)

    def __init__(self, name: str, inputs=(), outputs=(), state=None):
        assert all(isinstance(o, Output) for o in outputs)
        assert all(isinstance(i, Input) for i in inputs)
        assert state is None or isinstance(state, State)
        self._outputs = {o.name: o for o in outputs}
        self._inputs = {i.name: i for i in inputs}
        self._name = name
        self._local_state_indices = np.array([], dtype=int)
        self._state = state

    def compile(self):
        raise NotImplementedError

    def output_equation(self, output_name: str, numba_compile: bool = True):

        def wrapper(func):
            if numba_compile:
                output_dtype = self._outputs[output_name].dtype
                time_dtype = float_
                input_dtypes = tuple(inp.dtype for inp in self._outputs[output_name].system_inputs)
                if self._state is not None:
                    state_dtype = self._state.dtype
                    if len(input_dtypes) > 0:
                        signature = output_dtype(time_dtype, state_dtype, *input_dtypes)
                    else:
                        signature = output_dtype(time_dtype, state_dtype)
                else:
                    if len(input_dtypes) > 0:
                        signature = output_dtype(time_dtype, *input_dtypes)
                    else:
                        signature = output_dtype(time_dtype)
                func = nb.njit(signature)(func)
            self._outputs[output_name].output_equation = func

        return wrapper

    def state_equation(self, numba_compile: bool = True):
        assert self._state is not None

        def wrapper(func):
            if numba_compile:
                output_dtype = self._state.dtype
                time_dtype = float_
                state_dtype = self._state.dtype
                input_dtypes = tuple(inp.dtype for inp in self._state.system_inputs)
                signature = output_dtype(time_dtype, state_dtype, *input_dtypes)
                func = nb.njit(signature)(func)
            self._state.state_equation = func

        return wrapper
