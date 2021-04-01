import numba as nb
import numpy as np

from simba.types import float_array, float_

_registry = {}


def _register(stateful: bool, no_of_inputs: int, has_extra: bool):
    assert (stateful, no_of_inputs, has_extra) not in _registry.keys()

    def wrapper(func):
        _registry[(stateful, no_of_inputs, has_extra)] = func
        return func

    return wrapper


def create_output_function(
        output_equation, input_functions, local_state_indices, output_dtype, global_extra_type, extra_index
):
    stateful = len(local_state_indices) > 0
    has_extra = extra_index is not None
    try:
        fct = _registry[(stateful, len(input_functions), has_extra)](
            output_equation, input_functions, local_state_indices, extra_index
        )
    except KeyError:
        fct = _create_arbitrary_function(
            output_equation, input_functions, local_state_indices, extra_index
        )
    if type(output_equation) == nb.core.registry.CPUDispatcher\
            and all(type(in_fct) == nb.core.registry.CPUDispatcher for in_fct in input_functions):
        signature = output_dtype(float_, float_array, global_extra_type)
        fct = nb.njit(signature)(fct)
    return fct


@_register(False, 0, False)
def _create_stateless_0_input_fct(output_equation, input_functions, local_state_indices, extra_index):
    def mapping(t, global_state, global_extras):
        return output_equation(t)
    return mapping


@_register(False, 1, False)
def _create_stateless_1_input_fct(output_equation, input_functions, local_state_indices, extra_index):

    input_fct_0 = input_functions[0]

    def mapping(t, global_state, global_extras):
        input_0 = input_fct_0(t, global_state, global_extras)
        return output_equation(t, input_0)
    return mapping


@_register(False, 2, False)
def _create_stateless_2_input_fct(output_equation, input_functions, local_state_indices, extra_index):

    input_fct_0 = input_functions[0]
    input_fct_1 = input_functions[1]

    def mapping(t, global_state, global_extras):
        input_0 = input_fct_0(t, global_state, global_extras)
        input_1 = input_fct_1(t, global_state, global_extras)
        return output_equation(t, input_0, input_1)
    return mapping


@_register(False, 3, False)
def _create_stateless_3_input_fct(output_equation, input_functions, local_state_indices, extra_index):

    input_fct_0 = input_functions[0]
    input_fct_1 = input_functions[1]
    input_fct_2 = input_functions[2]

    def mapping(t, global_state, global_extras):
        input_0 = input_fct_0(t, global_state, global_extras)
        input_1 = input_fct_1(t, global_state, global_extras)
        input_2 = input_fct_2(t, global_state, global_extras)
        return output_equation(t, input_0, input_1, input_2)
    return mapping


@_register(True, 0, False)
def _create_stateful_0_input_fct(output_equation, input_functions, local_state_indices: np.ndarray, extra_index):

    def mapping(t, global_state: np.ndarray, extras):
        return output_equation(t, global_state[local_state_indices])
    return mapping


@_register(True, 1, False)
def _create_stateful_1_input_fct(output_equation, input_functions, local_state_indices, extra_index):

    input_function_0 = input_functions[0]

    def mapping(t, global_state, global_extras):
        local_state = global_state[local_state_indices]
        input_0 = input_function_0(t, global_state, global_extras)
        return output_equation(t, local_state, input_0)
    return mapping


@_register(True, 2, False)
def _create_stateful_2_input_fct(output_equation, input_functions, local_state_indices, extra_index):

    input_function_0 = input_functions[0]
    input_function_1 = input_functions[1]

    def mapping(t, global_state, global_extras):
        local_state = global_state[local_state_indices]
        input_0 = input_function_0(t, global_state, global_extras)
        input_1 = input_function_1(t, global_state, global_extras)
        return output_equation(t, local_state, input_0, input_1)
    return mapping


@_register(True, 3, False)
def _create_stateful_3_input_fct(output_equation, input_functions, local_state_indices):
    input_function_0 = input_functions[0]
    input_function_1 = input_functions[1]
    input_function_2 = input_functions[2]

    def mapping(t, global_state, global_extras):
        local_state = global_state[local_state_indices]
        input_0 = input_function_0(t, global_state, global_extras)
        input_1 = input_function_1(t, global_state, global_extras)
        input_2 = input_function_2(t, global_state, global_extras)
        return output_equation(t, local_state, input_0, input_1, input_2)
    return mapping


def _create_arbitrary_function(output_equation, input_functions, local_state_indices, extra_index):

    def spacing(no_of_spaces):
        return ' ' * no_of_spaces

    prior = ""
    header = "def mapping(t, global_state, global_extras):\n"

    if len(local_state_indices) > 0:
        state_reader = spacing(1) + "local_state = global_state[local_state_indices]\n"
        state_signature = " local_state,"
    else:
        state_reader = ""
        state_signature = ""

    input_reader = ""
    input_signature = ""
    for i in range(len(input_functions)):
        prior += f"input_function_{i} = input_functions[{i}]\n"
        input_reader += spacing(1) + f'input_{i} = input_function_{i}(t, global_state, global_extras)\n'
        input_signature += f" input_{i}, "
    if extra_index is not None:
        extras_reader = spacing(1) + f"extra = global_extras[extra_index]\n"
        extras_signature = "extra, "
    else:
        extras_reader = ""
        extras_signature = ""

    return_line = spacing(1) + f"return output_equation(t,{state_signature}{extras_signature}{input_signature})\n"
    appendix = "result.append(mapping)"
    fct = prior + header + state_reader + input_reader + extras_reader + return_line + appendix
    f = []
    exec(
        fct,
        {
            'result': f,
            'extra_index': extra_index,
            'output_equation': output_equation,
            'input_functions': input_functions
        }
    )
    return f[0]
