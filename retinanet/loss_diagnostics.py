import numpy as np


class InflectionDetector:
    def __init__(self, name, threshold):
        self.name = name
        self.threshold = threshold

        self._data = []
        self._grads = None
        self._diffs = None

    def is_value_anomalous(self, value):
        result = False
        self._data += [value]

        if len(self._data) > 1:
            self._grads = np.gradient(np.gradient(self._data))
            self._diffs = np.round(np.abs(np.diff(self._grads)), 3)
            result = self._diffs[-1] > self.threshold

        return result

    @property
    def data(self):
        return self._data
