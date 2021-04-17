import numba as nb

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


def _create_arbitrary_function(output_equation, input_functions, local_state_indices, extra_data_index):
    """
        exec(result, extra_data_index, output_equation, input_functions, local_state_indices):

            # Write to local variables to speed up numba computation significantly
            # input_function_{i} = input_functions[{i}]
            input_function_0 = input_functions[0]
            input_function_1 = input_functions[1]
            # ...

            def mapping(t, global_state, global_derivatives, global_extras):

                # input_{i} = input_function_{i}(t, global_state, global_extra_data)
                input_0 = input_function_0(t, global_state, global_extra_data)
                input_1 = input_function_1(t, global_state, global_extra_data)
                # ...

                # Read local state and extra data from the global data structures
                # Reading Extras and local state is optional, and executed only if the component has a state and extra
                # data
                local_state = global_state[local_state_indices]
                extra_data = global_extra_data[extra_data_index]

                return output_equation(t, local_state, extra_data, input_0, input_1)

            # Append the generated function to the (empty) result list to pass it back to the caller
            result.append(mapping)
        """

    def spacing(no_of_spaces):
        return ' ' * no_of_spaces

    prior = ""
    header = "def mapping(t, global_state, global_extra_data):\n"

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
        input_reader += spacing(1) + f'input_{i} = input_function_{i}(t, global_state, global_extra_data)\n'
        input_signature += f" input_{i}, "
    if extra_data_index is not None:
        extras_reader = spacing(1) + f"extra = global_extra_data[extra_data_index]\n"
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
            'extra_data_index': extra_data_index,
            'output_equation': output_equation,
            'input_functions': input_functions,
            'local_state_indices': local_state_indices
        }
    )
    return f[0]
