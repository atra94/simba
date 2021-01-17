import numba as nb

from .input import Input
from simba.core.function_factories.output_function_factory import create_output_function


class Output:

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self):
        return self._size

    @property
    def compiled(self):
        return self._output_function is not None

    @property
    def output_function(self):
        return self._output_function

    @property
    def system_inputs(self):
        return self._system_inputs

    @property
    def component(self):
        return self._component

    @property
    def external_inputs(self):
        return self._external_inputs

    @property
    def output_equation(self):
        return self._output_equation

    @output_equation.setter
    def output_equation(self, equation):
        self._output_equation = equation

    def __init__(self, component, name, system_inputs, size, signal_names=None, units='any', dtype=nb.types.Array(nb.float32, 1, 'C')):
        self._component = component
        self._size = size
        self._name = name
        self._units = units
        self._dtype = dtype
        self._external_inputs = set()
        self._system_inputs = system_inputs
        self._signal_names = signal_names
        self._output_equation = None
        self._global_state_indices = None
        self._output_function = None
        self._compiled = False

    def __call__(self, external_input):
        self.connect(external_input)

    def connect(self, input_: Input):
        self._external_inputs.add(input_)
        if input_.external_output != self:
            input_.connect(self)

    def disconnect(self, external_input):
        if external_input in self._external_inputs:
            self._external_inputs.remove(external_input)
            external_input.disconnect(self)

    def compile(self):
        if self._compiled:
            return
        assert self._output_equation is not None, 'Output equation has to be set before compilation.'
        for input_ in self._system_inputs:
            input_.compile()
        output_equation = self._output_equation
        input_functions = tuple([sys_input.function for sys_input in self._system_inputs])

        local_state_slice = self._component.local_state_slice
        self._output_function = create_output_function(
            output_equation, input_functions, local_state_slice, self._dtype
        )
        self._compiled = True

        """ Future Content
        if len(self.derivative_equations) != len(self._system_inputs):
            return
        derivative_equations = self._derivative_equations
        input_derivative_functions = tuple(input_.derivative_function for input_ in self._system_inputs)

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
