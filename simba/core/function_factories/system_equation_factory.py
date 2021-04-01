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


@_register(1)
def _create_1_state_fct(state_functions, state_length):

    state_function_0 = state_functions[0]

    def mapping(t, global_state, global_extras):
        global_derivatives = np.zeros(state_length)
        state_function_0(t, global_state, global_derivatives, global_extras)
        return global_derivatives
    return mapping


@_register(2)
def _create_2_state_fct(state_functions, state_length):
    state_function_0 = state_functions[0]
    state_function_1 = state_functions[1]

    def mapping(t, global_state, global_extras):
        global_derivatives = np.zeros(state_length)
        state_function_0(t, global_state, global_derivatives, global_extras)
        state_function_1(t, global_state, global_derivatives, global_extras)
        return global_derivatives
    return mapping


@_register(3)
def _create_3_state_fct(state_functions, state_length):
    state_function_0 = state_functions[0]
    state_function_1 = state_functions[1]
    state_function_2 = state_functions[2]

    def mapping(t, global_state, global_extras):
        global_derivatives = np.zeros(state_length)
        state_function_0(t, global_state, global_derivatives, global_extras)
        state_function_1(t, global_state, global_derivatives, global_extras)
        state_function_2(t, global_state, global_derivatives, global_extras)
        return global_derivatives

    return mapping


@_register(4)
def _create_4_state_fct(state_functions, state_length):
    state_function_0 = state_functions[0]
    state_function_1 = state_functions[1]
    state_function_2 = state_functions[2]
    state_function_3 = state_functions[3]

    def mapping(t, global_state, global_extras):
        global_derivatives = np.zeros(state_length)
        state_function_0(t, global_state, global_derivatives, global_extras)
        state_function_1(t, global_state, global_derivatives, global_extras)
        state_function_2(t, global_state, global_derivatives, global_extras)
        state_function_3(t, global_state, global_derivatives, global_extras)
        return global_derivatives
    return mapping


@_register(5)
def _create_5_state_fct(state_functions, state_length):
    state_function_0 = state_functions[0]
    state_function_1 = state_functions[1]
    state_function_2 = state_functions[2]
    state_function_3 = state_functions[3]
    state_function_4 = state_functions[4]

    def mapping(t, global_state, global_extras):
        global_derivatives = np.zeros(state_length)
        state_function_0(t, global_state, global_derivatives, global_extras)
        state_function_1(t, global_state, global_derivatives, global_extras)
        state_function_2(t, global_state, global_derivatives, global_extras)
        state_function_3(t, global_state, global_derivatives, global_extras)
        state_function_4(t, global_state, global_derivatives, global_extras)
        return global_derivatives

    return mapping


@_register(6)
def _create_6_state_fct(state_functions, state_length):
    state_function_0 = state_functions[0]
    state_function_1 = state_functions[1]
    state_function_2 = state_functions[2]
    state_function_3 = state_functions[3]
    state_function_4 = state_functions[4]
    state_function_5 = state_functions[5]

    def mapping(t, global_state, global_extras):
        global_derivatives = np.zeros(state_length)
        state_function_0(t, global_state, global_derivatives, global_extras)
        state_function_1(t, global_state, global_derivatives, global_extras)
        state_function_2(t, global_state, global_derivatives, global_extras)
        state_function_3(t, global_state, global_derivatives, global_extras)
        state_function_4(t, global_state, global_derivatives, global_extras)
        state_function_5(t, global_state, global_derivatives, global_extras)
        return global_derivatives
    return mapping


@_register(7)
def _create_7_state_fct(state_functions, state_length):
    state_function_0 = state_functions[0]
    state_function_1 = state_functions[1]
    state_function_2 = state_functions[2]
    state_function_3 = state_functions[3]
    state_function_4 = state_functions[4]
    state_function_5 = state_functions[5]
    state_function_6 = state_functions[6]

    def mapping(t, global_state, global_extras):
        global_derivatives = np.zeros(state_length)
        state_function_0(t, global_state, global_derivatives, global_extras)
        state_function_1(t, global_state, global_derivatives, global_extras)
        state_function_2(t, global_state, global_derivatives, global_extras)
        state_function_3(t, global_state, global_derivatives, global_extras)
        state_function_4(t, global_state, global_derivatives, global_extras)
        state_function_5(t, global_state, global_derivatives, global_extras)
        state_function_6(t, global_state, global_derivatives, global_extras)
        return global_derivatives

    return mapping


@_register(8)
def _create_8_state_fct(state_functions, state_length):
    state_function_0 = state_functions[0]
    state_function_1 = state_functions[1]
    state_function_2 = state_functions[2]
    state_function_3 = state_functions[3]
    state_function_4 = state_functions[4]
    state_function_5 = state_functions[5]
    state_function_6 = state_functions[6]
    state_function_7 = state_functions[7]

    def mapping(t, global_state, global_extras):
        global_derivatives = np.zeros(state_length)
        state_function_0(t, global_state, global_derivatives, global_extras)
        state_function_1(t, global_state, global_derivatives, global_extras)
        state_function_2(t, global_state, global_derivatives, global_extras)
        state_function_3(t, global_state, global_derivatives, global_extras)
        state_function_4(t, global_state, global_derivatives, global_extras)
        state_function_5(t, global_state, global_derivatives, global_extras)
        state_function_6(t, global_state, global_derivatives, global_extras)
        state_function_7(t, global_state, global_derivatives, global_extras)
        return global_derivatives

    return mapping


def _create_arbitrary_function(state_functions, state_length):
    def mapping():
        return None

    def spacing(no_of_spaces):
        return ' ' * no_of_spaces

    state_reader = ""
    header = "def mapping(t, global_state, global_extras):\n"
    state_executions = ""
    derivatives = spacing(1) + f"global_derivatives = np.zeros(state_length)\n"
    return_line = "return global_derivatives"

    for i in range(len(state_functions)):
        state_executions += spacing(1) + f"state_function_{i}(t, global_state, global_derivatives, global_extras)\n"
        state_reader += f"state_function_{i} = state_functions[{i}]\n"

    fct = state_reader + header + derivatives + state_executions + return_line

    exec(fct)

    return mapping
