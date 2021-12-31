import numpy as np
from typing import List
from .input import Input
import simba as sb
import simba.core.function_factories.output_function_factory as off
from simba.core.interfaces import ISliceHolder
from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from simba.core import Input, SystemComponent


class Output(ISliceHolder):

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def size(self) -> int:
        return self._size

    @property
    def compiled(self) -> bool:
        return self._output_function is not None

    @property
    def output_function(self) -> Callable or None:
        return self._output_function

    @property
    def component_inputs(self) -> List['Input']:
        return self._component_inputs

    @property
    def component(self) -> 'SystemComponent':
        return self._component

    @property
    def external_inputs(self) -> Iterable['Input']:
        return self._external_inputs

    @property
    def output_equation(self) -> Callable:
        return self._output_equation

    @output_equation.setter
    def output_equation(self, equation: Callable):
        self._output_equation = equation

    def __init__(self, component: 'SystemComponent', name: str, component_inputs: Iterable['Input'], size: int,
                 signal_names: Iterable[str] or None = None, units='any', dtype: type = sb.types.float_base_type):
        ISliceHolder.__init__(self)
        self._component = component
        self._size = size
        self._name = name
        self._units = units
        self._dtype = dtype
        self._external_inputs = set()
        self._component_inputs = component_inputs
        self._signal_names = signal_names
        self._output_equation = None
        self._output_function = None
        self._compiled = False

    def __call__(self, external_input: 'Input'):
        self.connect(external_input)

    def connect(self, input_: 'Input'):
        self._external_inputs.add(input_)
        if input_.external_output != self:
            input_.connect(self)

    def disconnect(self, external_input: 'Input'):
        if external_input in self._external_inputs:
            self._external_inputs.remove(external_input)
            external_input.disconnect(self)

    def compile(self, global_extra_type):
        assert self._output_equation is not None, 'Output equation has to be set before compilation.'
        assert self._local_slice is not None, 'Local slice has to be set before compilation.'
        self._output_function = off.create_output_function(self, global_extra_type)
        self._compiled = True

        """ Future Content
        if len(self.derivative_equations) != len(self._component_inputs):
            return
        derivative_equations = self._derivative_equations
        input_derivative_functions = tuple(input_.derivative_function for input_ in self._component_inputs)

        def derivative_function(t, global_state, derivative_array, derivative_product):
            local_state = global_state[state_indices]
            derivatives = tuple(derivative(t, local_state) for derivative in derivative_equations)
            next_derivative_product = np.matmul(derivative_product, derivatives)
            # local_derivative = state_derivative(t, local_state)
            # derivative_array[local_state, :] = np.matmul(local_derivative, derivative_product)
            for i in range(len(derivatives)):
                input_derivative_functions[i](t, global_state, derivative_array, next_derivative_product)

        self._derivative_function = derivative_function
        """
