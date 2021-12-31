import numpy as np
import numba as nb

float_base_type = nb.float64

float_array = float_base_type[:]

int_base_type = nb.int32

int_array = int_base_type[:]


def array_type(base_type):
    return base_type[:]


def initial_value(type_, size):
    if type_ is float_base_type:
        return 0.0
    elif type_ is float_array:
        return np.zeros(size, dtype=float)
    elif type_ is int_base_type:
        return 0
    elif type_ is int_array:
        return np.zeros(size, dtype=int)
