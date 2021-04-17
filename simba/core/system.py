import numba as nb
import numpy as np

from simba.core.function_factories.system_equation_factory import create_system_equation
from simba.basic_components import Logger
from simba.core.system_components import SystemInput, SystemOutput


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

    @property
    def extras(self):
        return self._extras

    @property
    def system_input(self):
        return self._system_input

    @property
    def system_output(self):
        return self._system_output

    def __init__(self, components, system_inputs=(), system_outputs=()):
        components_ = [component for component in components if not isinstance(component, Logger)]
        loggers_ = [logger for logger in components if isinstance(logger, Logger)]
        namelist = [component.name for component in components]
        assert len(set(namelist)) == len(namelist), 'Duplicate names in the components. Use all unique names.'
        self._compiled = False
        self._system_equation = None
        self._extras = ()

        self._system_input = SystemInput(system_inputs)
        components_.append(self._system_input)

        self._system_output = SystemOutput(system_outputs)
        components_.append(self._system_output)

        self._components = {component.name: component for component in components_}
        self._loggers = {logger.name: logger for logger in loggers_}
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

    def set_input(self, inputs):
        self._system_input.set_input(inputs)

    def get_output(self, t, state):
        return self._system_output(t, state, self._extras)

    def compile(self, numba_compile=True):

        current_extra_idx = [0]
        extras = []

        def get_extra_index(extra):
            extras.append(extra)
            current_extra_idx[0] = current_extra_idx[0] + 1
            return current_extra_idx[0] - 1

        start_index = 0
        for state in self._states.values():
            state.local_state_slice = np.arange(start_index, start_index + state.size)
            start_index += state.size
        for component in self._components.values():
            component.compile(get_extra_index, numba_compile=numba_compile)
        self._extras = tuple(extras)
        extra_types = ()
        for extra_ in self._extras:
            extra_types = extra_types + (nb.typeof(extra_),)
        global_extra_type = nb.types.Tuple(extra_types)
        for output in self._outputs.values():
            output.compile(global_extra_type)
        for input_ in self._inputs.values():
            input_.compile(global_extra_type)
        for state in self._states.values():
            state.compile(global_extra_type)
        state_functions = tuple([state.state_function for state in self._states.values()])
        state_length = self._state_length

        system_equation = create_system_equation(state_functions, state_length, global_extra_type)

        self._system_equation = lambda t, global_state: system_equation(t, global_state, self._extras)
