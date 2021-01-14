from .output import Output
from .input import Input
from .state import State


class SystemComponent:

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def name(self):
        return self._name

    def __init__(self, name: str, inputs=(), outputs=()):

        assert all(isinstance(o, Output) for o in outputs)
        assert all(isinstance(i, Input) for i in inputs)
        self._outputs = {o.name: o for o in outputs}
        self._inputs = {i.name: i for i in inputs}
        self._name = name


class StatefulSystemComponent(SystemComponent):

    @property
    def state(self):
        return self._state

    def __init__(self, state: State, inputs=None, outputs=None):
        super().__init__(inputs, outputs)
        assert isinstance(state, State)
        self._state = state
