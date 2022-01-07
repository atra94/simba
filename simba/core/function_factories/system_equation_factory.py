import numba as nb
import numpy as np

from simba.types import float_base_type, float_array, int_array


def create_system_equation(state_functions, output_functions, state_length, float_length, int_length, global_extra_type):

    fct = _create_arbitrary_function(state_functions, output_functions, state_length, float_length, int_length)
    if all(type(fct_) == nb.core.registry.CPUDispatcher for fct_ in list(state_functions) + list(output_functions)):
        signature = float_array(float_base_type, float_array, global_extra_type)
        fct = nb.njit(signature)(fct)
    return fct


def _create_arbitrary_function(state_functions, output_functions, state_length, float_length, int_length):
    """
    exec(f, state_length, state_functions, f, np):

        # Write to local variables to speed up numba computation significantly
        # output_function_{i} = output_functions[{i}]
        output_function_0 = output_functions[0]
        # ...

        # state_function_{i} = state_functions[{i}]
        state_function_0 = state_functions[0]
        state_function_1 = state_functions[1]
        # ...

        def system_equation(t, global_state, global_float_outputs, global_int_outputs, global_extras):
            global_derivatives = np.zeros(state_length)

            # Call each output function to fill the output vectors
            # output_function_{i}(t, global_state, global_float_outputs, global_int_outputs, global_extras)
            output_function_0(t, global_state, global_float_outputs, global_int_outputs, global_extras)
            output_function_1(t, global_state, global_float_outputs, global_int_outputs, global_extras)
            # ...

            # Call each State function separately
            # state_function_{i}(
            #     t, global_state, global_float_outputs, global_int_outputs, global_derivatives, global_extras
            # )

            state_function_0(
                 t, global_state, global_float_outputs, global_int_outputs, global_derivatives, global_extras
            )
            state_function_1(
                 t, global_state, global_float_outputs, global_int_outputs, global_derivatives, global_extras
            )
            # ...

        # Append the generated function to the (empty) list f to pass it back to the caller
        f.append(mapping)
    """

    def spacing(no_of_spaces):
        return ' ' * no_of_spaces

    state_reader = ""
    output_reader = ""
    header = "def system_equation(t, global_state, global_extras):\n"
    state_function_calls = ""
    output_function_calls = ""
    derivatives = spacing(1) + f"global_derivatives = np.zeros(state_length)\n"
    derivatives += spacing(1) + f"global_float_outputs = np.zeros(float_length)\n"
    derivatives += spacing(1) + f"global_int_outputs = np.zeros(int_length, dtype=np.int32)\n"
    return_line = spacing(1) + "return global_derivatives\n"
    for i in range(len(output_functions)):
        output_function_calls += spacing(1) \
            + f"output_function_{i}(t, global_state, global_float_outputs, global_int_outputs, global_extras)\n"
        output_reader += f"output_function_{i} = output_functions[{i}]\n"
    for i in range(len(state_functions)):
        state_function_calls += spacing(1) \
            + f"state_function_{i}(" \
              "t, global_state, global_float_outputs, global_int_outputs, global_derivatives, global_extras" \
              ")\n"
        state_reader += f"state_function_{i} = state_functions[{i}]\n"
    appendix = "result.append(system_equation)\n"
    system_equation_code = state_reader \
        + output_reader \
        + header \
        + derivatives \
        + output_function_calls \
        + state_function_calls \
        + return_line \
        + appendix
    f = []
    exec(
        system_equation_code,
        {
            'state_length': state_length,
            'float_length': float_length,
            'int_length': int_length,
            'state_functions': state_functions,
            'output_functions': output_functions,
            'result': f,
            'np': np,
        }
    )
    return f[0]

