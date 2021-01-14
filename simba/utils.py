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
