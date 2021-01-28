""" Hyperparameters for trainer and model. """
import json
import collections.abc
from typing import Text, List, Dict

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

    @property
    def params(self) -> EasyDict:
        return self._params

    @property
    def keys(self) -> List:
        return self.config_dict.keys()

    @classmethod
    def _parser(cls, config: Text) -> Dict:
        config_dict = json.loads(config)
        return config_dict

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
        assert isinstance(config, str), "wrong type for hparams"
        try:
            config_dict = self._parser(config)
        except:
            raise ValueError("wrong extension or format for hparams")
        self._params = self.update(self._params, config_dict)
        return self._params
