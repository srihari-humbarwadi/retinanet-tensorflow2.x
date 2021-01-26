import os
import copy
import unittest

from retinanet.cfg.config import Config

class TestConfig(unittest.TestCase):
    def setUp(self):
        _sample_config = ['configs', 'v3-32', 'retinanet-640-6x-256-tpu-pod.json']
        self.config_path = os.path.join(*_sample_config)
        self.train_override = 100
        self.sample_hparams_override = '''{"training":{"batch_size":{"train":100}}}'''
        self._config = Config(self.config_path)

    def test_config_override(self):
        params = self._config.override(self.sample_hparams_override)
        assert params.training.batch_size.train == self.train_override
