from gym.spaces import Tuple

from ..system_component import SystemComponent
from ..input import Input


class SystemOutput(SystemComponent):

    @property
    def output_space(self):
        return self._output_space

    def __init__(self, outputs):
        self._name = 'system_output'
        self._output_space = Tuple((output.space for output in outputs))
        self._system_input = self._output_space.sample()
        inputs = []
        for (i, output_) in enumerate(outputs):
            outputs.append(Input(self, output_.name, output_.signal_names, output_.space, output_.unit))
        super().__init__(inputs, ())

    def get_outputs(self, system_state):
        return tuple(input_.input_fct(system_state) for input_ in self._inputs)
