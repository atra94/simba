import numpy as np
import simba as sb

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simba.core import Output


class Input:

    @property
    def default_value(self):
        return self._default_value

    @default_value.setter
    def default_value(self, value):
        self._default_value = np.asarray(value)
        assert self.size == self._default_value.size

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        val = int(value)
        if val == self._size:
            return
        self._size = val
        if self._external_output is None:
            return
        if self._external_output.size is None:
            self._external_output.size = val
        assert self._external_output.size == val, \
            f'Size Mismatch: Input {self._component.name}.{self.name} has size {self.size} and ' \
            f'Output {self._external_output.component.name}.{self._external_output.name} ' \
            f'has size {self._external_output.size}.'

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def name(self) -> str:
        return self._name

    @property
    def component(self):
        return self._component

    @property
    def external_output(self) -> 'Output':
        return self._external_output

    @property
    def connected(self) -> bool:
        return self._external_output is not None

    def __init__(self, component, name: str, size: int, default_value=None, dtype=float):
        # (SystemComponent):  Overlying System Component of the Input
        self._component = component

        # (str): Inputs identifying name
        self._name = name

        # (int / None): Space of the Input. None for unset until connection.
        self._size = size

        # (type / None):
        self._dtype = dtype

        # (Output): The output from another system that the input is connected to.
        self._external_output = None

        # (np.ndarray / None): Default value to be used if no Output is connected.
        # If a default value is specified, a dtype and a size have to be specified during initialization.
        # None: A connected output is required.
        self._default_value = None
        if default_value is not None:
            self.default_value = default_value

    def __call__(self, output):
        self.connect(output)

    def connect(self, output):
        if self._external_output is not None:
            self._external_output.disconnect(self)
        if not isinstance(output, sb.core.Output):
            self.default_value = output
            return
        self._external_output = output
        assert output.dtype == self.dtype, \
            f'Datatype Mismatch: Input dtype {self._dtype}, ' \
            f'Output dtype: {output.dtype}. Connection aborted'
        assert output.size == self._size, \
            f'Size Mismatch: Input size {self._size}, Output size: {output.size}. Connection aborted'
        if self not in output.external_inputs:
            output.connect(self)

    def disconnect(self, output):
        if output == self._external_output:
            self._external_output = None
            output.disconnect(self)
