import numba as nb
import numpy as np

from simba.types import float_array, float_

_registry = {}


def _register(stateful: bool, no_of_inputs: int):
    assert (stateful, no_of_inputs) not in _registry.keys()

    def wrapper(func):
        _registry[(stateful, no_of_inputs)] = func
        return func

    return wrapper


def create_output_function(output_equation, input_functions, local_state_slice, output_dtype):
    stateful = len(local_state_slice) > 0
    try:
        fct = _registry[(stateful, len(input_functions))](output_equation, input_functions, local_state_slice)
    except KeyError:
        fct = _create_arbitrary_function(output_equation, input_functions, local_state_slice)
    if type(output_equation) == nb.core.registry.CPUDispatcher\
            and all(type(in_fct) == nb.core.registry.CPUDispatcher for in_fct in input_functions):
        signature = output_dtype(float_, float_array)
        fct = nb.njit(signature)(fct)
    return fct


@_register(False, 0)
def _create_stateless_0_input_fct(output_equation, input_functions, local_state_slice):
    def mapping(t, global_state):
        return output_equation(t)
    return mapping


@_register(False, 1)
def _create_stateless_1_input_fct(output_equation, input_functions, local_state_slice):

    input_fct_0 = input_functions[0]

    def mapping(t, global_state):
        input_0 = input_fct_0(t, global_state)
        return output_equation(t, input_0)

    return mapping


@_register(False, 2)
def _create_stateless_2_input_fct(output_equation, input_functions, local_state_slice):

    input_fct_0 = input_functions[0]
    input_fct_1 = input_functions[1]

    def mapping(t, global_state):
        input_0 = input_fct_0(t, global_state)
        input_1 = input_fct_1(t, global_state)
        return output_equation(t, input_0, input_1)
    return mapping


@_register(False, 3)
def _create_stateless_3_input_fct(output_equation, input_functions, local_state_slice):

    input_fct_0 = input_functions[0]
    input_fct_1 = input_functions[1]
    input_fct_2 = input_functions[2]

    def mapping(t, global_state):
        input_0 = input_fct_0(t, global_state)
        input_1 = input_fct_1(t, global_state)
        input_2 = input_fct_2(t, global_state)
        return output_equation(t, input_0, input_1, input_2)
    return mapping


@_register(True, 0)
def _create_stateful_0_input_fct(output_equation, input_functions, local_state_slice: np.ndarray):

    def mapping(t, global_state: np.ndarray):
        return output_equation(t, global_state[local_state_slice])
    return mapping


@_register(True, 1)
def _create_stateful_1_input_fct(output_equation, input_functions, local_state_slice):

    input_function_0 = input_functions[0]

    def mapping(t, global_state):
        local_state = global_state[local_state_slice]
        input_0 = input_function_0(t, global_state)
        return output_equation(t, local_state, input_0)
    return mapping


@_register(True, 2)
def _create_stateful_2_input_fct(output_equation, input_functions, local_state_slice):

    input_function_0 = input_functions[0]
    input_function_1 = input_functions[1]

    def mapping(t, global_state):
        local_state = global_state[local_state_slice]
        input_0 = input_function_0(t, global_state)
        input_1 = input_function_1(t, global_state)
        return output_equation(t, local_state, input_0, input_1)
    return mapping


@_register(True, 3)
def _create_stateful_3_input_fct(output_equation, input_functions, local_state_slice):
    input_function_0 = input_functions[0]
    input_function_1 = input_functions[1]
    input_function_2 = input_functions[2]

    def mapping(t, global_state):
        local_state = global_state[local_state_slice]
        input_0 = input_function_0(t, global_state)
        input_1 = input_function_1(t, global_state)
        input_2 = input_function_2(t, global_state)
        return output_equation(t, local_state, input_0, input_1, input_2)
    return mapping


def _create_arbitrary_function(output_equation, input_functions, local_state_slice: slice):
    def mapping():
        return None

    def spacing(no_of_spaces):
        return ' ' * no_of_spaces

    header = "def mapping(t, global_state):\n"

    if local_state_slice != slice(0):
        state_reader = spacing(1) + "local_state = global_state[local_state_slice]\n"
        state_signature = " local_state,"
    else:
        state_reader = ""
        state_signature = ""

    input_reader = ""
    input_signature = ""
    for i in range(len(input_functions)):
        input_reader += spacing(1) + f'input_{i} = input_functions[{i}](t, global_state)\n'
        input_signature += " input_{i},"

    return_line = spacing(1) + f"return output_equation(t,f{state_signature}f{input_signature})"
    fct = header + state_reader + input_reader + return_line
    exec(fct)
    return mapping
