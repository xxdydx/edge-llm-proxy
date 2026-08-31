"""A minimal least-recently-used cache."""


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self._data = {}
        self._order = []  # least-recently-used first

    def put(self, key, value):
        if key in self._data:
            self._order.remove(key)
        elif len(self._data) >= self.capacity:
            oldest = self._order.pop(0)
            del self._data[oldest]
        self._data[key] = value
        self._order.append(key)

    def get(self, key):
        """Return the cached value for `key`, or None if it is not present."""
        return self._data.get(key)

    def __len__(self):
        return len(self._data)
