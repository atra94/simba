from gym.spaces import Tuple

from ..system_component import SystemComponent
from ..output import Output


class SystemInput(SystemComponent):

    @property
    def input_space(self):
        return self._input_space

    def __init__(self, inputs):
        self._name = 'system_input'
        self._input_space = Tuple((input_.space for input_ in inputs))
        self._system_input = self._input_space.sample()
        outputs = []
        for (i, input_) in enumerate(inputs):
            def output_eq(*_):
                return self._system_input[i]
            outputs.append(Output(self, input_.name, output_eq, (), input_.signal_names, input_.space, input_.unit))
        super().__init__((), outputs)

    def set_input(self, system_input):
        assert system_input in self._input_space
        self._system_input = system_input
