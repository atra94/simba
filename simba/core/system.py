import numpy as np


class System:

    @property
    def compiled(self):
        return self._compiled

    @property
    def system_equation(self):
        assert self._compiled, 'The system has to be compiled before accessing the system equation.'
        return self._system_equation

    @property
    def components(self):
        return self._components

    @property
    def state_length(self):
        return self._state_length

    def __init__(self, components):
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

        for component in self._components.values():
            component.compile(numba_compile=numba_compile)
        for state in self._states.values():
            state.local_state_indices = slice(start_index, start_index+state.size)
            start_index += state.size
            state.compile()
        for output in self._outputs.values():
            output.compile()
        for input_ in self._inputs.values():
            input_.compile()

        state_functions = [state.state_function for state in self._states.values()]

        def system_equation(t, y):
            derivatives = np.zeros(self._state_length, dtype=float)
            for state_function in state_functions:
                derivatives = state_function(t, y, derivatives)
            return derivatives

        self._system_equation = system_equation