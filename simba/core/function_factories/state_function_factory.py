import numba as nb
import numpy as np

from simba.types import float_array, float_

_registry = {}


def _register(no_of_inputs: int, has_extra: bool):
    assert no_of_inputs not in _registry.keys()

    def wrapper(func):
        _registry[(no_of_inputs, has_extra)] = func
        return func

    return wrapper


def create_state_function(state_equation, input_functions, local_state_indices, global_extra_type, extra_index):
    has_extra = extra_index is not None
    try:
        fct = _registry[len(input_functions), has_extra](
            state_equation, input_functions, local_state_indices, extra_index
        )
    except KeyError:
        fct = _create_arbitrary_function(state_equation, input_functions, local_state_indices, extra_index)
    if type(state_equation) == nb.core.registry.CPUDispatcher\
            and all(type(in_fct) == nb.core.registry.CPUDispatcher for in_fct in input_functions):
        signature = nb.none(float_, float_array, float_array, global_extra_type)
        fct = nb.njit(signature)(fct)
    return fct


def _create_arbitrary_function(state_equation, input_functions, local_state_indices, extra_index):
    """
        exec(result, extra_index, state_equation, input_functions, local_state_indices):

            # Write to local variables to speed up numba computation significantly
            # input_function_{i} = input_functions[{i}]
            input_function_0 = input_functions[0]
            input_function_1 = input_functions[1]
            # ...

            def mapping(t, global_state, global_derivatives, global_extras):

                # input_{i} = input_function_{i}(t, global_state, global_extras)
                input_0 = input_function_0(t, global_state, global_extras)
                input_1 = input_function_1(t, global_state, global_extras)
                # ...

                # Read local state and extra data from the global data structures
                local_state = global_state[local_state_indices]
                extra = global_extras[extra_index]

                global_derivatives[local_state_indices] = state_equation(t, local_state, extra, input_0, input_1)

            # Append the generated function to the (empty) result list to pass it back to the caller
            result.append(mapping)
        """
    def spacing(no_of_spaces):
        return ' ' * no_of_spaces

    header = "def mapping(t, global_state, global_derivatives, global_extras):\n"
    state_reader = spacing(1) + "local_state = global_state[local_state_indices]\n"
    state_signature = " local_state,"

    input_reader = ""
    input_signature = ""
    prior = ""
    for i in range(len(input_functions)):
        input_reader += spacing(1) + f'input_{i} = input_function_{i}(t, global_state, global_extras)\n'
        input_signature += f" input_{i},"
        prior += f"input_function_{i} = input_functions[{i}]\n"

    if extra_index is not None:
        extra_reader = spacing(1) + f"extra =  global_extras[extra_index]\n"
        extra_signature = "extra, "
    else:
        extra_signature = ""
        extra_reader = ""

    return_line = spacing(1) + f"global_derivatives[local_state_indices] = " \
        f"state_equation(t,{state_signature}{extra_signature}{input_signature})\n"

    appendix = "result.append(mapping)\n"
    fct = prior + header + state_reader + input_reader + extra_reader + return_line + appendix
    f = []
    exec(
        fct,
        {
            'result': f,
            'extra_index': extra_index,
            'state_equation': state_equation,
            'input_functions': input_functions,
            'local_state_indices': local_state_indices,
        }
    )
    return f[0]
