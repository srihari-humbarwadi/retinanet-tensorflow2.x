import json
from abc import ABC, abstractmethod

from absl import logging


class Parser(ABC):
    def __init__(self, download_path, name='Parser'):
        self._name = '_'.join(name.lower().split())
        self._download_path = download_path
        self._data = {'train': [], 'val': []}
        self._classes = set()
        self._class_name_to_class_id = {}
        self._class_id_to_class_name = {}

    def get_class_id(self, class_name=None):
        return self._class_name_to_class_id[class_name]

    def get_class_name(self, class_id=None):
        return self._class_id_to_class_name[class_id]

    def dump_label_map(self):
        logging.info('Dumping label map for {} dataset'.format(self._name))

        with open(self._name + '_label_map.json', 'w') as f:
            json.dump(self._class_id_to_class_name, f, indent=4)

    def dump_parsed_json(self):
        logging.info('Dumping parsed json for {} dataset'.format(self._name))

        with open(self._name + '_parsed_dataset.json', 'w') as f:
            parsed_dataset = {'name': self._name, 'dataset': self._data}
            json.dump(parsed_dataset, f, indent=4)

    def dump_parsed_dataset(self):
        self.dump_label_map()
        self.dump_parsed_json()

    @abstractmethod
    def _build_dataset(self):
        pass

    @property
    def name(self):
        return self._name

    @property
    def dataset(self):
        return self._data

    @property
    def classes(self):
        return self._classes
