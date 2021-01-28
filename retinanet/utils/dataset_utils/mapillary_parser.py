import json
import os

from absl import logging
from tqdm import tqdm

from retinanet.dataset_utils.parser import Parser


class MapillaryParser(Parser):

    def __init__(self,
                 download_path,
                 image_ext='.jpg',
                 discard_classes=False,
                 only_val=False,
                 skip_ambiguous=False,
                 name='Mapillary Traffic Sign'):
        super(MapillaryParser, self).__init__(download_path, name=name)

        self._image_ext = '.jpg'
        self._only_val = only_val
        self._skip_ambiguous = skip_ambiguous
        self._discard_classes = discard_classes
        self._annotation_dir = os.path.join(download_path, 'annotations')
        self._splits_dir = os.path.join(download_path, 'splits')
        self._images_dir = os.path.join(download_path, 'images')

        self._ambiguous_instances = {'train': 0, 'val': 0}
        self._skipped_samples = {'train': 0, 'val': 0}
        self._skipped_annotations = {'train': 0, 'val': 0}

        self._splits = self._load_splits()
        self._build_dataset()

    def _load_splits(self):
        logging.info('Loading splits')
        splits = {}
        for split_name in ['train', 'val']:
            with open(os.path.join(self._splits_dir, split_name + '.txt'), 'r') as f:
                splits[split_name] = [line.strip() for line in f.readlines()]
        return splits

    def _build_dataset(self):
        if self._discard_classes:

            self._class_name_to_class_id['traffic_sign'] = 1
            self._classes.add('traffic_sign')
            logging.warning('Mapping all classses to `traffic_sign`')

        def _is_box_valid(box, image_height, image_width):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                return False

            if x1 + width > image_width or y1 + height > image_height:
                return False

            return True

        def _build(split_name):
            logging.info('Parsing {} split from {} dataset'.format(
                split_name, self._name))

            image_names = self._splits[split_name]

            for _idx, image_name in tqdm(
                    enumerate(sorted(image_names)), total=len(image_names)):
                image_path = os.path.join(
                    self._images_dir,
                    image_name + self._image_ext)

                annotation_path = os.path.join(
                    self._annotation_dir, image_name + '.json')

                with open(annotation_path, 'r') as f:
                    annotation = json.load(f)

                image_height = annotation['height']
                image_width = annotation['width']

                boxes = []
                classes = []

                for obj in annotation['objects']:
                    box = [
                        obj['bbox']['xmin'] / image_width,
                        obj['bbox']['ymin'] / image_height,
                        obj['bbox']['xmax'] / image_width,
                        obj['bbox']['ymax'] / image_height
                    ]
                    class_name = obj['label']

                    if class_name not in self._classes and not self._discard_classes:
                        self._class_name_to_class_id[class_name] = \
                            len(self._classes) + 1
                        self._classes.add(class_name)

                    if self._skip_ambiguous and obj['properties']['ambiguous']:
                        self._ambiguous_instances[split_name] += 1
                        continue

                    if not _is_box_valid(box,
                                         image_height,
                                         image_width):
                        self._skipped_annotations[split_name] += 1
                        continue

                    boxes.append(box)
                    class_id_encoded = \
                        1 if self._discard_classes else self.get_class_id(class_name)
                    classes.append(class_id_encoded)

                if len(classes) == 0:
                    self._skipped_samples[split_name] += 1
                    continue

                sample = {
                    'image': image_path,
                    'image_id': int(_idx),
                    'image_height': image_height,
                    'image_width': image_width,
                    'label': {
                        'boxes': boxes,
                        'classes': classes
                    }
                }
                self._data[split_name].append(sample)

        if not self._only_val:
            _build('train')
        _build('val')

        self._class_id_to_class_name = {
            v: k
            for k, v in self._class_name_to_class_id.items()
        }

        for split_name in ['train', 'val']:
            logging.info(
                'Successfully parsed {} {} samples from {} dataset'.format(
                    len(self._data[split_name]), split_name, self._name))

            logging.info('Skipped {} {} empty samples'.format(
                self._skipped_samples[split_name], split_name))

            logging.info('Skipped {} {} bad annotations'.format(
                self._skipped_annotations[split_name], split_name))

            if self._skip_ambiguous:
                logging.info(
                    'Skipped {} ambiguous instances instances from {} samples'
                    .format(self._ambiguous_instances[split_name], split_name))
