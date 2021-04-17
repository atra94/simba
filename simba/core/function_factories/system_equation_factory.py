import numba as nb
import numpy as np

from simba.types import float_, float_array

_registry = {}


def _register(no_of_states: int):
    assert no_of_states not in _registry.keys()

    def wrapper(func):
        _registry[no_of_states] = func
        return func

    return wrapper


def create_system_equation(state_functions, state_length, global_extra_type):
    try:
        fct = _registry[len(state_functions)](state_functions, state_length)
    except KeyError:
        fct = _create_arbitrary_function(state_functions, state_length)
    if all(type(state_fct) == nb.core.registry.CPUDispatcher for state_fct in state_functions):
        signature = float_array(float_, float_array, global_extra_type)
        fct = nb.njit(signature)(fct)
    return fct


def _create_arbitrary_function(state_functions, state_length):
    """
    exec(f, state_length, state_functions, f, np):

        # Write to local variables to speed up numba computation significantly
        # state_function_{i} = state_functions[{i}]
        state_function_0 = state_functions[0]
        state_function_1 = state_functions[1]
        # ...

        def mapping(t, global_state, global_extras):
            global_derivatives = np.zeros(state_length)

            # Call each State function separately
            # state_function_{i}(t, global_state, global_derivatives, global_extras)

            state_function_0(t, global_state, global_derivatives, global_extras)
            state_function_1(t, global_state, global_derivatives, global_extras)
            # ...

        # Append the generated function to the (empty) list f to pass it back to the caller
        f.append(mapping)
    """

    def spacing(no_of_spaces):
        return ' ' * no_of_spaces

    state_reader = ""
    header = "def mapping(t, global_state, global_extras):\n"
    state_executions = ""
    derivatives = spacing(1) + f"global_derivatives = np.zeros(state_length)\n"
    return_line = spacing(1) + "return global_derivatives\n"

    for i in range(len(state_functions)):
        state_executions += spacing(1) + f"state_function_{i}(t, global_state, global_derivatives, global_extras)\n"
        state_reader += f"state_function_{i} = state_functions[{i}]\n"
    appendix = "result.append(mapping)\n"
    fct = state_reader + header + derivatives + state_executions + return_line + appendix
    f = []
    exec(
        fct,
        {
            'state_length': state_length,
            'state_functions': state_functions,
            'result': f,
            'np': np,
        }
    )
    return f[0]

