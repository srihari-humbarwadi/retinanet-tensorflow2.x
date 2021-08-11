import numpy as np


class InflectionDetector:
    def __init__(self, name, threshold, skip_steps=45):
        if skip_steps < 2:
            raise ValueError('`skip_steps` should be greater than 2')

        self.name = name
        self.threshold = threshold

        self._skip_steps = skip_steps
        self._data = []
        self._grads = None
        self._diffs = None

    def is_value_anomalous(self, value):
        result = False
        self._data += [value]

        if len(self._data) > self._skip_steps:
            self._grads = np.gradient(np.gradient(self._data))
            self._diffs = np.round(np.abs(np.diff(self._grads)), 3)
            result = self._diffs[-2] > self.threshold

        return result

    def reset(self):
        self._data = []
        self._grads = None
        self._diffs = None

    @property
    def data(self):
        return self._data
