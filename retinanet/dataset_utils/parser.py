import json
from abc import ABC, abstractmethod

from absl import logging
from tqdm import tqdm


class Parser(ABC):
    def __init__(self, download_path, name='Parser', remap_class_ids=False):
        self._name = '_'.join(name.lower().split())
        self._download_path = download_path
        self._remap_class_ids = remap_class_ids
        self._data = {'train': [], 'val': []}
        self._classes = set()
        self._class_name_to_class_id = {}
        self._class_id_to_class_name = {}
        self._remapping_info = {}

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

    def dump_remapping_info(self):
        logging.info('Dumping remapping info for {} dataset'.format(self._name))

        with open(self._name + '_remapping_info.json', 'w') as f:
            json.dump(self._remapping_info, f, indent=4)

    def dump_parsed_dataset(self):
        self.dump_label_map()
        self.dump_parsed_json()
        self.dump_remapping_info()

    def _remap(self):
        orig_ids = list(self._class_id_to_class_name.keys())
        logging.info('Remapping class ids')

        sorted_classes = sorted(self._classes)
        sorted_class_name_to_class_id = {
            class_name: idx for idx, class_name in enumerate(sorted_classes)
            }
        sorted_class_id_to_class_name = {
            idx: class_name for idx, class_name in enumerate(sorted_classes)
            }
        orig_class_id_to_remapped_class_id = {
            old_idx: sorted_class_name_to_class_id[class_name]
            for old_idx, class_name in self._class_id_to_class_name.items()
            }
        remapped_class_id_to_orig_class_id = {
            v: k for k, v in orig_class_id_to_remapped_class_id.items()
            }

        for split in self._data.keys():
            if self._data[split] == []:
                continue

            logging.info('Remapping {} split'.format(split))
            for i in tqdm(range(len(self._data[split]))):
                sample = self._data[split][i]
                old_class_ids = sample['label']['classes']
                new_class_ids = [orig_class_id_to_remapped_class_id[idx]
                                 for idx in old_class_ids]
                sample['label']['classes'] = new_class_ids
                self._data[split][i] = sample

        self._class_name_to_class_id = sorted_class_name_to_class_id
        self._class_id_to_class_name = sorted_class_id_to_class_name

        self._remapping_info = {
            'sorted_classes': sorted_classes,
            'class_name_to_class_id': sorted_class_name_to_class_id,
            'class_id_to_class_name': sorted_class_id_to_class_name,
            'orig_class_id_to_remapped_class_id': orig_class_id_to_remapped_class_id,
            'remapped_class_id_to_orig_class_id': remapped_class_id_to_orig_class_id
        }
        remapped_ids = list(self._class_id_to_class_name.keys())
        logging.info(
            'Successfully remapped {} classes with class ids ranging from '
            '[{}-{}] to [{}-{}]'.format(
                len(self._classes),
                min(orig_ids),
                max(orig_ids),
                min(remapped_ids),
                max(remapped_ids)))

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
