import numba as nb
import numpy as np


class System:

    @property
    def compiled(self):
        return self._compiled

    @property
    def system_equation(self):
        assert self._system_equation is not None, 'The system has to be compiled before accessing the system equation.'
        return self._system_equation

    @property
    def components(self):
        return self._components

    @property
    def state_length(self):
        return self._state_length

    def __init__(self, components):
        namelist = [component.name for component in components]
        assert len(set(namelist)) == len(namelist), 'Duplicate names in the components. Use all unique names.'
        self._compiled = False
        self._system_equation = None

        self._components = {component.name: component for component in components}
        self._inputs = dict()
        self._outputs = dict()
        self._states = dict()
        for component in self._components.values():
            for output in component.outputs.values():
                self._outputs[f'{component.name}.{output.name}'] = output
            for input_ in component.inputs.values():
                self._inputs[f'{component.name}.{input_.name}'] = input_
            if component.state is not None:
                self._states[f'{component.name}.State'] = component.state

        self._stateful_components = tuple(
            component for component in components if component.state is not None
        )
        self._state_length = sum(state.size for state in self._states.values())

    def _order_outputs(self):
        ordered_outputs = []
        remaining_outputs = self._outputs[:]
        for output in self._outputs:
            if len(output.system_inputs) == 0:
                ordered_outputs.append(output)
                remaining_outputs.remove(output)
        i = 0
        while len(remaining_outputs) > 0:
            i = (i + 1) % len(remaining_outputs)
            output = remaining_outputs[i]
            if all(input_.connected_output in ordered_outputs for input_ in output.system_inputs):
                ordered_outputs.append(output)
                remaining_outputs.remove(output)

        self._outputs = ordered_outputs

    def compile(self, numba_compile=True):
        start_index = 0
        for state in self._states.values():
            state.local_state_slice = np.arange(start_index, start_index + state.size)
            start_index += state.size
        for component in self._components.values():
            component.compile(numba_compile=numba_compile)
        for output in self._outputs.values():
            output.compile()
        for input_ in self._inputs.values():
            input_.compile()
        for state in self._states.values():
            state.compile()
        state_functions = tuple([state.state_function for state in self._states.values()])
        state_length = self._state_length
        signature = nb.types.Array(nb.float32, 1, 'C')(nb.float32, nb.types.Array(nb.float32, 1, 'C'))
        state_fct_0 = state_functions[0]
        state_fct_1 = state_functions[1]

        def system_equation(t, y):
            derivatives = np.zeros(state_length, dtype=np.float32)
            state_fct_0(t, y, derivatives)
            state_fct_1(t, y, derivatives)
            return derivatives

        if numba_compile:
            system_equation = nb.njit(signature)(system_equation)
        self._system_equation = system_equation
