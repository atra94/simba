import numpy as np
import numba as nb

float_ = nb.float64

float_array = nb.float64[:]

int_ = nb.int32

int_array = nb.int32[:]


def initial_value(type_, size):
    if type_ is float_:
        return 0.0
    elif type_ is float_array:
        return np.zeros(size, dtype=float)
    elif type_ is int_:
        return 0
    elif type_ is int_array:
        return np.zeros(size, dtype=int)
