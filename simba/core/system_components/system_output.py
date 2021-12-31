from gym.spaces import Tuple

from ..system_component import SystemComponent
from ..input import Input


class SystemOutput(SystemComponent):

    @property
    def output_sizes(self):
        return self._output_sizes

    @property
    def output_dtypes(self):
        return self._output_dtypes

    def __init__(self, outputs):
        self._output_sizes = tuple(output.size for output in outputs)
        self._output_dtypes = tuple(output.dtype for output in outputs)
        inputs = []
        for (i, output_) in enumerate(outputs):
            input_ = Input(self, output_.name, output_.size, dtype=output_.dtype)
            input_.connect(output_)
            inputs.append(input_)
        super().__init__('system_output', inputs, ())

    def __call__(self, t, global_state, global_extras):
        return self.get_outputs(t, global_state, global_extras)

    def get_outputs(self, t, global_state, global_extras):
        return tuple(input_.function(t, global_state, global_extras) for input_ in self._inputs.values())

    def compile(self, get_extra_indices, numba_compile=True):
        pass
