""" Hyperparameters for trainer and model. """
import json

from absl import logging
from easydict import EasyDict


class Config:
    """ Configuration for hparams class. """

    def __init__(self, path):
        self.path = path
        self._load()

    def _load(self):
        logging.info('Loading config from {}'.format(self.path))
        with open(self.path, 'r') as fp:
            self._params = EasyDict(json.load(fp))

    @property
    def params(self):
        return self._params

    @property
    def keys(self):
        return self.config_dict.keys()

    @classmethod
    def _parser(cls, config):
        config_dict = json.loads(config)
        return config_dict

    def override(self, config: str):
        assert isinstance(config, str), "wrong type for hparams"
        try:
            config_dict = self._parser(config)
        except:
            raise ValueError("wrong extension or format for hparams")
        self._params.update(config_dict)
        return self._params
