import numba as nb

_registry = {}


def _register(no_of_inputs: int):
    assert no_of_inputs not in _registry.keys()

    def wrapper(func):
        _registry[no_of_inputs] = func
        return func

    return wrapper


def create_state_function(state_equation, input_functions, local_state_indices):
    try:
        fct = _registry[len(input_functions)](state_equation, input_functions, local_state_indices)
    except KeyError:
        fct = _create_arbitrary_function(state_equation, input_functions, local_state_indices)
    if type(state_equation) == nb.core.registry.CPUDispatcher\
            and all(type(in_fct) == nb.core.registry.CPUDispatcher for in_fct in input_functions):
        signature = nb.none(nb.float32, nb.types.Array(nb.float32, 1, 'C'), nb.types.Array(nb.float32, 1, 'C'))
        fct = nb.njit(signature)(fct)
    return fct


@_register(0)
def _create_0_input_fct(state_equation, input_functions, local_state_indices):

    def mapping(t, global_state, global_derivatives):
        local_state = global_state[local_state_indices]
        global_derivatives[local_state_indices] = state_equation(t, local_state)

    return mapping


@_register(1)
def _create_1_input_fct(state_equation, input_functions, local_state_indices):

    input_function_0 = input_functions[0]

    def mapping(t, global_state, global_derivatives):
        local_state = global_state[local_state_indices]
        input_0 = input_function_0(t, global_state)
        global_derivatives[local_state_indices] = state_equation(t, local_state, input_0)
    return mapping


@_register(2)
def _create_2_input_fct(state_equation, input_functions, local_state_indices):

    input_function_0 = input_functions[0]
    input_function_1 = input_functions[1]

    def mapping(t, global_state, global_derivatives):
        local_state = global_state[local_state_indices]
        input_0 = input_function_0(t, global_state)
        input_1 = input_function_1(t, global_state)
        global_derivatives[local_state_indices] = state_equation(t, local_state, input_0, input_1)
    return mapping


@_register(3)
def _create_3_input_fct(state_equation, input_functions, local_state_indices):
    def mapping(t, global_state, global_derivatives):
        local_state = global_state[local_state_indices]
        input_0 = input_functions[0](t, global_state)
        input_1 = input_functions[1](t, global_state)
        input_2 = input_functions[2](t, global_state)
        global_derivatives[local_state_indices] = state_equation(t, local_state, input_0, input_1, input_2)
    return mapping


def _create_arbitrary_function(state_equation, input_functions, local_state_indices):
    def mapping():
        return None

    def spacing(no_of_spaces):
        return ' ' * no_of_spaces

    header = "def mapping(t, global_state, global_derivatives):\n"
    state_reader = spacing(1) + "local_state = global_state[local_state_indices]\n"
    state_signature = " local_state,"

    input_reader = ""
    input_signature = ""
    for i in range(len(input_functions)):
        input_reader += spacing(1) + f'input_{i} = input_functions[{i}](t, global_state)\n'
        input_signature += " input_{i},"

    result_line = \
        spacing(1) + f"global_derivatives[local_state_indices = state_equation(t,f{state_signature}f{input_signature})"
    fct = header + state_reader + input_reader + result_line
    exec(fct)
    return mapping
