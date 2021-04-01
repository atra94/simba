import numba as nb

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


@_register(0, False)
def _create_0_input_fct(state_equation, input_functions, local_state_indices, _):

    def mapping(t, global_state, global_derivatives, global_extras):
        local_state = global_state[local_state_indices]
        global_derivatives[local_state_indices] = state_equation(t, local_state)

    return mapping


@_register(1, False)
def _create_1_input_fct(state_equation, input_functions, local_state_indices, _):

    input_function_0 = input_functions[0]

    def mapping(t, global_state, global_derivatives, global_extras):
        local_state = global_state[local_state_indices]
        input_0 = input_function_0(t, global_state, global_extras)
        global_derivatives[local_state_indices] = state_equation(t, local_state, input_0)
    return mapping


@_register(2, False)
def _create_2_input_fct(state_equation, input_functions, local_state_indices, _):

    input_function_0 = input_functions[0]
    input_function_1 = input_functions[1]

    def mapping(t, global_state, global_derivatives, global_extras):
        local_state = global_state[local_state_indices]
        input_0 = input_function_0(t, global_state, global_extras)
        input_1 = input_function_1(t, global_state, global_extras)
        global_derivatives[local_state_indices] = state_equation(t, local_state, input_0, input_1)
    return mapping


@_register(3, False)
def _create_3_input_fct(state_equation, input_functions, local_state_indices, _):
    def mapping(t, global_state, global_derivatives, global_extras):
        local_state = global_state[local_state_indices]
        input_0 = input_functions[0](t, global_state, global_extras)
        input_1 = input_functions[1](t, global_state, global_extras)
        input_2 = input_functions[2](t, global_state, global_extras)
        global_derivatives[local_state_indices] = state_equation(t, local_state, input_0, input_1, input_2)
    return mapping


def _create_arbitrary_function(state_equation, input_functions, local_state_indices, extra_index):
    def mapping():
        return None

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
        input_signature += " input_{i},"
        prior += f"input_function_{i} = input_functions[{i}]\n"

    if extra_index is not None:
        extra_reader = spacing(1) + f"extra =  global_extras[extra_index]\n"
        extra_signature = "extra, "
    else:
        extra_signature = ""
        extra_reader = ""

    result_line = spacing(1) + f"global_derivatives[local_state_indices] = " \
        f"state_equation(t,{state_signature}{extra_signature}{input_signature})"
    fct = prior + header + state_reader + extra_reader + input_reader + result_line
    exec(fct)
    return mapping
