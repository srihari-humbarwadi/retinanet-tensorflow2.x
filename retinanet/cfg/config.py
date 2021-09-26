import json

from absl import logging
from easydict import EasyDict
import tensorflow as tf


class Config:
    def __init__(self, path):
        self.path = path
        self._load()

    def _load(self):
        logging.info('Loading config from {}'.format(self.path))
        with tf.io.gfile.GFile(self.path, 'r') as fp:
            self._params = EasyDict(json.load(fp))
        logging.debug('\n' + json.dumps(self._params, indent=4))

    @property
    def params(self):
        return self._params
