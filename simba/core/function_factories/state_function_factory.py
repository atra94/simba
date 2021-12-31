import numba as nb
import numpy as np

import simba as sb
from simba.types import float_array, float_base_type, int_array, int_base_type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simba.core import State


def create_state_function(state, global_extra_type):

    fct = _create_state_function(state)
    if type(state.state_equation) == nb.core.registry.CPUDispatcher:
        signature = nb.none(float_base_type, float_array, float_array, int_array, float_array, global_extra_type)
        fct = nb.njit(signature)(fct)
    return fct


def _create_state_function(state: 'State'):
    """
        exec(result, state):

            # Write to local variables to speed up numba computation significantly
            # input_function_{i} = input_functions[{i}]
            input_slice_0 = state.component_inputs[0].external_output.local_output_slice
            input_1 = state.component_inputs[1].default_value
            # ...

            def state_function(
                t, global_state, global_float_outputs, global_int_outputs, global_derivatives, global_extras
            ):

                # input_{i} = input_function_{i}(t, global_state, global_extras)
                input_0 = global_float_output[input_slice_0]

                # ...

                # Read local state and extra data from the global data structures
                local_state = global_state[local_state_indices]
                extra = global_extras[extra_index]

                global_derivatives[local_state_indices] = state_equation(t, local_state, extra, input_0, input_1)

            # Append the generated function to the (empty) result list to pass it back to the caller
            result.append(state_function)
        """
    def spacing(no_of_spaces):
        return ' ' * no_of_spaces

    header = "def state_function(t, global_state, global_float_outputs, global_int_outputs, global_derivatives," \
             " global_extras):\n"
    state_reader = spacing(1) + "local_state = global_state[local_state_indices]\n"
    state_signature = " local_state,"

    input_reader = ""
    input_signature = ""
    prior = ""
    prior += "local_state_indices = state.local_slice\n"
    prior += "state_equation = state.state_equation\n"
    for i, input_ in enumerate(state.component_inputs):
        if input_.connected:
            prior += f"input_slice_{i} = state.component_inputs[{i}].external_output.local_slice\n"
            if input_.dtype == float_base_type:
                arr = 'global_float_outputs'
            elif input_.dtype == int_base_type:
                arr = 'global_int_outputs'
            else:
                raise AttributeError(f'Illegal dtype of input {input_.name}: {input_.dtype}. Must be float or int')
            input_reader += spacing(1) + f'input_{i} = {arr}[input_slice_{i}]\n'
        else:
            prior += f'input_{i} = output.component_inputs[{i}].default_value\n'

        input_signature += f" input_{i}, "

    if state.component.extra_index is not None:
        extra_reader = spacing(1) + f"extra =  global_extras[extra_index]\n"
        extra_signature = "extra, "
    else:
        extra_signature = ""
        extra_reader = ""

    return_line = spacing(1) + f"global_derivatives[local_state_indices] = " \
        f"state_equation(t,{state_signature}{extra_signature}{input_signature})\n"

    appendix = "result.append(state_function)\n"
    fct = prior + header + state_reader + input_reader + extra_reader + return_line + appendix
    f = []
    exec(
        fct,
        {
            'result': f,
            'state': state
        }
    )
    return f[0]
