class IContext:

    @property
    def name(self):
        return self._name

    @property
    def context(self):
        return self._context

    @property
    def path(self):
        return self._context.path + f'.{self._name}' if self._context.path is not '' else self._name

    def __init__(self, context, name):
        assert type(name) is str
        self._name = name

        assert isinstance(context, IContext)
        self._context = context

        self._path = self._context.path + f'.{name}'
