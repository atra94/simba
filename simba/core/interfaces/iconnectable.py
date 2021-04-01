class IConnectable:

    @property
    def size(self) -> int:
        return self._size

    @property
    def dtype(self) -> type:
        return self._dtype

    def __init__(self, size, dtype):
        assert type(size) is int and size > 0
        assert type(dtype) is type

        self._size = size
        self._dtype = dtype

    def connect(self, other):
        raise NotImplementedError

    def disconnect(self, other):
        raise NotImplementedError
