from ..output import Output
from ..system_component import SystemComponent
from ...types import initial_value


class SystemInput(SystemComponent):

    @property
    def input_sizes(self):
        return self._input_sizes

    @property
    def input_dtypes(self):
        return self._input_dtypes

    def __init__(self, inputs):
        self._input_sizes = (input_.size for input_ in inputs)
        self._input_dtypes = (input_.dtype for input_ in inputs)
        outputs = []
        for (i, input_) in enumerate(inputs):
            output = Output(
                self, input_.name, (), input_.size, input_.signal_names, input_.unit, input_.accepted_dtypes[0]
            )
            output.connect(input_)

        super().__init__('system_input', (), outputs)

    def set_input(self, system_inputs, global_extras):
        for system_input, output in zip(self._outputs, system_inputs):
            extra = global_extras[output.extra_index]
            extra[:] = system_input

    def compile(self, get_extra_index, numba_compile=True):
        for output in self._outputs:
            output_extra = initial_value(output.dtype, output.size)
            output.extra_index = get_extra_index(output_extra)

            @self.output_equation(output.name, numba_compile=numba_compile)
            def output_equation(t, extra):
                return extra
