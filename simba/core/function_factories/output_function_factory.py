import numba as nb
import simba as sb
from simba.types import float_array, float_base_type, int_array, int_base_type


def create_output_function(output, global_extra_type):
    fct = _create_arbitrary_function(output)
    if type(output.output_equation) == nb.core.registry.CPUDispatcher:
        signature = nb.none(float_base_type, float_array, float_array, int_array, global_extra_type)
        fct = nb.njit(signature)(fct)
    return fct


def _create_arbitrary_function(output):
    """
        exec(result, output):

            output_equation = output.output_equation

            # Write to local variables to speed up numba computation significantly
            # input_function_{i} = input_functions[{i}]
            input_slice_0 = output.component_inputs[0].external_output.local_slice
            input_1 = output.component_inputs[1].default_value
            input_slice_2 = output.component_inputs[0].external_output.local_slice
            # ...

            def output_function(t, global_state, global_float_outputs, global_int_outputs, global_extras):

                # input_{i} = input_function_{i}(t, global_state, global_extra_data)
                input_0 = global_float_outputs[input_slice_0]
                input_2 = global_int_outputs[input_slice_2]
                # ...

                # Read local state and extra data from the global data structures
                # Reading Extras and local state is optional, and executed only if the component has a state and extra
                # data
                local_state = global_state[local_state_indices]
                extra_data = global_extra_data[extra_data_index]

                global_float_outputs[local_slice] = output_equation(t, local_state, extra_data, input_0,input_1,input_2)

            # Append the generated function to the (empty) result list to pass it back to the caller
            result.append(output_function)
        """

    def spacing(no_of_spaces):
        return ' ' * no_of_spaces

    prior = ""
    prior += "output_equation = output.output_equation\n"
    prior += "output_slice = output.local_slice\n"
    header = "def output_function(t, global_state, global_float_outputs, global_int_outputs, global_extra_data):\n"
    state = output.component.state
    if state is not None:
        prior += "local_state_indices = output.component.state.local_slice\n"
        state_reader = spacing(1) + "local_state = global_state[local_state_indices]\n"
        state_signature = " local_state,"
    else:
        state_reader = ""
        state_signature = ""

    input_reader = ""
    input_signature = ""
    for i, input_ in enumerate(output.component_inputs):
        if input_.connected:
            prior += f"input_slice_{i} = output.component_inputs[{i}].external_output.local_slice\n"
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
    if output.component.extra_index is not None:
        prior += "extra_data_index = output.component.extra_index\n"
        extras_reader = spacing(1) + f"extra = global_extra_data[extra_data_index]\n"
        extras_signature = "extra, "
    else:
        extras_reader = ""
        extras_signature = ""

    #return_line = spacing(1) + f"return output_equation(t,{state_signature}{extras_signature}{input_signature})\n"
    if output.dtype == float_base_type:
        target = 'global_float_outputs'
    elif output.dtype == int_base_type:
        target = 'global_int_outputs'
    else:
        raise AssertionError(f'Illegal output dtype: {output.dtype}')
    writer = spacing(1)\
        + f'{target}[output_slice] = output_equation(t,{state_signature}{extras_signature}{input_signature})\n'
    appendix = "result.append(output_function)"
    fct = prior + header + state_reader + input_reader + extras_reader + writer + appendix
    f = []
    exec(
        fct,
        {
            'result': f,
            'output': output
        }
    )
    return f[0]
