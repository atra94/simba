from typing import Iterable, List, TYPE_CHECKING
import simba as sb
from collections import deque

if TYPE_CHECKING:
    from simba.core import Output


class Event:
    
    def __init__(self):
        self._callee_list = set()

    def __iadd__(self, fct):
        return self._callee_list.add(fct)

    def __isub__(self, fct):
        return self._callee_list.remove(fct)

    def __call__(self, sender, *event_args):
        for callee in self._callee_list:
            callee(sender, *event_args)


def sort_outputs_old(outputs: Iterable['Output']) -> List['Output']:
    input_degrees = dict()
    ordering = []
    next_outputs = deque()
    for output in outputs:
        degree = len([comp_input for comp_input in output.component_inputs if comp_input.connected])
        input_degrees[output] = degree
        if input_degrees[output] == 0:
            next_outputs.append(output)
    while len(next_outputs) > 0:
        output = next_outputs.popleft()
        ordering.append(output)
        for successor in output.external_inputs:
            input_degrees[successor] -= 1
            if input_degrees[successor] == 0:
                next_outputs.append(successor)
    return ordering


def sort_outputs(outputs: Iterable['Output']) -> List['Output']:
    remaining_outputs = list(outputs)
    ordering = []

    while len(remaining_outputs) > 0:
        initial_length = len(remaining_outputs)
        for output in remaining_outputs:
            predecessors = [inp.external_output for inp in output.component_inputs if inp.connected]
            if all([predecessor in ordering for predecessor in predecessors]):
                remaining_outputs.remove(output)
                ordering.append(output)
        assert len(remaining_outputs) < initial_length, 'Circular dependency in the calculation.'
    return ordering
