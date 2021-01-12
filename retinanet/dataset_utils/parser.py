from abc import ABC, abstractmethod


class Parser(ABC):
    def __init__(self, download_path):
        self._download_path = download_path
        self._data = {'train': [], 'val': []}
        self._classes = set()
        self._class_name_to_class_id = {}
        self._class_id_to_class_name = {}

    def get_class_id(self, class_name=None):
        return self._class_name_to_class_id[class_name]

    def get_class_name(self, class_id=None):
        return self._class_id_to_class_name[class_id]

    @abstractmethod
    def _build_dataset(self):
        pass

    @property
    def dataset(self):
        return self._data

    @property
    def classes(self):
        return self._classes
