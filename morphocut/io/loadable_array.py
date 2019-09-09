import numpy as np


class LoadableArray(np.ndarray):
    @staticmethod
    def load(loader, id, index=None):
        arr = loader(id, index)
        return LoadableArray(arr, loader, id, index)

    def __new__(cls, input_array, loader, id, index):
        obj = np.asarray(input_array).view(cls)

        obj.loader = loader
        obj.id = id
        obj.index = index

        obj.setflags(write=False)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.loader = getattr(obj, 'loader', None)
        self.id = getattr(obj, 'id', None)
        self.index = getattr(obj, 'index', None)

    def __reduce__(self):
        return LoadableArray.load, (self.loader, self.id, self.index)

    def __getitem__(self, index):
        if self.index is not None:
            raise NotImplementedError("Unable to calculate a slice of a slice")
        result = super().__getitem__(index)
        result.index = index
        return result

    def __repr__(self):
        return repr(self.view(np.ndarray))
