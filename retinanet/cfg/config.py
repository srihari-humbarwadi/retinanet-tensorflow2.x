""" Hyperparameters for trainer and model. """
import collections.abc
import json
from typing import List, Text

from absl import logging
from easydict import EasyDict


class Config:
    """ Configuration for hparams class. """

    def __init__(self, path: Text):
        self.path = path
        self._load()

    def _load(self):
        logging.info('Loading config from {}'.format(self.path))
        with open(self.path, 'r') as fp:
            self._params = EasyDict(json.load(fp))
        logging.debug(self)

    def __repr__(self):
        return '\n' + json.dumps(self._params, indent=4)

    @property
    def params(self) -> EasyDict:
        return self._params

    @property
    def keys(self) -> List:
        return self._params.keys()

    @staticmethod
    def update(d, u):
        def _update(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = _update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        return _update(d, u)

    def override(self, config: Text) -> EasyDict:

        if not isinstance(config, str):
            raise AssertionError(
                '`config` must be a json string but got {}'.format(type(config)))

        try:
            config_dict = json.loads(config)

        except json.JSONDecodeError:
            raise ValueError(
                'Failed to decode json string, please validate your config overides')

        self._params = self.update(self._params, config_dict)
        return self._params
