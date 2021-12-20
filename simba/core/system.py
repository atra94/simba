import numba as nb
import numpy as np

import simba as sb
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

        self._components = dict()
        self._inputs = dict()
        self._outputs = dict()
        self._states = dict()

        def add_single_components_io(
                component: sb.core.SystemComponent, output_dict, input_dict, state_dict, path: str
        ):
            for output in component.outputs.values():
                output_dict[f'{path}{component.name}.{output.name}'] = output
            for input_ in component.inputs.values():
                input_dict[f'{path}{component.name}.{input_.name}'] = input_
            if component.state is not None:
                state_dict[f'{path}{component.name}.State'] = component.state

        def add_components(components__, component_dict, output_dict, input_dict, state_dict, path):
            for component in components__:
                component_dict[f'{path}{component.name}'] = component
                add_single_components_io(component, output_dict, input_dict, state_dict, path)
                if isinstance(component, sb.basic_components.Subsystem):
                    add_components(
                        component.components, component_dict, output_dict, input_dict, state_dict,
                        f'{path}{component.name}.'
                    )

        add_components(components_, self._components, self._outputs, self._inputs, self._states, '')
        self._loggers = {logger.name: logger for logger in loggers_}

        self._stateful_components = tuple(
            component for component in self._components.values() if component.state is not None
        )
        self._state_length = sum(state.size for state in self._states.values())

    def set_input(self, inputs):
        self._system_input.set_input(inputs, self._extras)

    def get_output(self, t, state):
        return self._system_output(t, state, self._extras)

    def compile(self, numba_compile=True):

        current_extra_idx = [0]
        extras = []
        extra_types = []

        def get_extra_index(extra):
            extras.append(extra)
            extra_types.append(nb.typeof(extra))
            current_extra_idx[0] = current_extra_idx[0] + 1
            return current_extra_idx[0] - 1

        start_index = 0
        # Set the state slice indices of the components
        for state in self._states.values():
            state.local_state_slice = np.arange(start_index, start_index + state.size)
            start_index += state.size

        # Compile each component
        for component in self._components.values():
            component.compile(get_extra_index, numba_compile=numba_compile)

        # Compile the outputs
        for output in self._outputs.values():
            if output.caching_time is not None and output.cache_index is None:
                output.cache_index = current_extra_idx[0]
                cache = nb.float64([0.] * output.size)
                caching_time = nb.float64([-np.inf])
                extras.append(caching_time)
                extras.append(cache)
                extra_types.append(nb.typeof(caching_time))
                extra_types.append(nb.typeof(cache))
                current_extra_idx[0] += 2
        self._extras = tuple(extras)
        extra_types = tuple(extra_types)
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
