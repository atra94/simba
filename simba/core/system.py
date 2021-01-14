from .system_component import StatefulSystemComponent

import numba as nb
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

    def __init__(self, components, system_input=None):
        self._compiled = False
        self._system_equation = None

        self._components = components
        self._inputs = dict()
        self._outputs = dict()
        for component in self._components:
            self._outputs += component.outputs
            self._inputs += component.inputs

        self._stateful_components = tuple(
            component for component in self._components if isinstance(component, StatefulSystemComponent)
        )
        self._state_length = sum(component.state.state_length for component in self._stateful_components)

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

    def system_equation(self, t, y, *inputs):
        derivatives = np.zeros(self._state_length)
        for state in self._states:
            derivatives = state.state_function(t, y, derivatives, *inputs)
        return derivatives

    def compile(self):
        for input_ in self._inputs:
            assert input_.connected_output is not None, f'Input {input_.name} is unconnected.'
        self._order_outputs()
