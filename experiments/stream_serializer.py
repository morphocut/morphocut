from collections.abc import MutableMapping


class TwoLevelCache(MutableMapping):
    """
    ...
    """

    def __init__(self):
        self.lvl1 = {}
        self.lvl2 = {}
        self.lvl2_accessed = set()

    def swap(self):
        # 1: Remove elements from lvl1 that were not accessed (and clear lvl2_accessed)
        self.lvl2 = {k: self.lvl2[k] for k in self.lvl2_accessed}
        self.lvl2_accessed.clear()

        # 2: Move elements from lvl1 to lvl2 (and clear lvl1)
        self.lvl2.update(self.lvl1)
        self.lvl1.clear()

    def __getitem__(self, key):
        try:
            return self.lvl1[key]
        except KeyError:
            pass

        value = self.lvl2[key]
        self.lvl2_accessed.add(key)
        return value

    def __setitem__(self, key, value):
        ...

    def __delitem__(self, key):
        ...

    def __iter__(self):
        ...

    def __len__(self):
        return len(self.lvl1) + len(self.lvl2)
