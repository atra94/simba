import numpy as np


class Input:

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
    def accepted_dtypes(self) -> tuple:
        return self._accepted_dtypes

    @property
    def name(self) -> str:
        return self._name

    @property
    def component(self):
        return self._component

    @property
    def function(self):
        return self._function

    @property
    def compiled(self) -> bool:
        return self._function is not None

    @property
    def external_output(self):
        return self._external_output

    def __init__(self, component, name: str, size: int, accepted_dtypes: tuple, default_value=None):
        # (SystemComponent):  Overlying System Component of the Input
        self._component = component

        # (str): Inputs identifying name
        self._name = name

        # (int / None): Space of the Input. None for unset until connection.
        self._size = size

        # (type / None):
        self._dtype = None

        self._compiled = False

        self._accepted_dtypes = tuple(accepted_dtypes)

        # (Output): The output from another system that the input is connected to.
        self._external_output = None

        # (np.ndarray / None): Default value to be used if no Output is connected.
        # If a default value is specified, a dtype and a size have to be specified during initialization.
        # None: A connected output is required.
        if default_value is not None:
            self._default_value = np.asarray(default_value, dtype=self.dtype)
            assert size == self._default_value.size
        self._default_value = default_value

        # Set during the compilation.
        # It will be the compiled output_equation or a compiled default value, if unconnected.
        self._function = None

    def __call__(self, output):
        self.connect(output)

    def compile(self):
        if self._compiled:
            return
        assert self._external_output is not None or self._default_value is not None, \
            f'Unconnected Input {self._component.name}.{self._name}:' \
            f' Either an Output has to be connected to the input or a default value has to be set.'
        if self._external_output is not None:
            if not self._external_output.compiled:
                self._external_output.compile()
            self._function = self._external_output.output_function
        else:
            default_value = self._default_value
            self._function = lambda t, global_state: default_value
        self._compiled = True

    def connect(self, output):
        if self._external_output is not None:
            self._external_output.disconnect(self)
        self._external_output = output
        assert output.dtype in self._accepted_dtypes, \
            f'Datatype Mismatch: Accepted Input dtypes {self._accepted_dtypes}, ' \
            f'Output dtype: {output.dtype}. Connection aborted'
        assert output.size == self._size, \
            f'Size Mismatch: Input size {self._size}, Output size: {output.size}. Connection aborted'
        self._dtype = output.dtype
        if self not in output.external_inputs:
            output.connect(self)

    def disconnect(self, output):
        if output == self._external_output:
            self._external_output = None
            output.disconnect(self)
