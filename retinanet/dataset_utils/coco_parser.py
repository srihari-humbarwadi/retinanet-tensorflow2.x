import os

import numpy as np
from absl import logging
from pycocotools.coco import COCO
from tqdm import tqdm

from retinanet.dataset_utils.parser import Parser


class CocoParser(Parser):
    _NAME = 'COCO'
    _YEAR = '2017'

    def __init__(self,
                 download_path,
                 only_mappings=False,
                 only_val=False,
                 skip_crowd=True):
        super(CocoParser, self).__init__(download_path)
        self._only_mappings = only_mappings
        self._only_val = only_val
        self._skip_crowd = skip_crowd

        if not only_val:
            self._train_annotations_path = os.path.join(
                download_path, 'annotations/instances_train2017.json')
        self._val_annotations_path = os.path.join(
            download_path, 'annotations/instances_val2017.json')

        self._crowd_instances = {'train': 0, 'val': 0}
        self._skipped_samples = {'train': 0, 'val': 0}
        self._annotation = {}
        self._build_dataset()

    def _build_dataset(self):
        def _convert_box_format(boxes):
            boxes = np.array(boxes)
            return np.concatenate([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]],
                                  axis=-1)

        def _build(annotations_path, split_name):
            logging.info('Parsing {} split from {} dataset'.format(
                split_name, CocoParser._NAME))
            coco = COCO(annotations_path)
            if self._class_id_to_class_name == {}:
                self._class_id_to_class_name = {
                    cat_dict['id']: cat_dict['name']
                    for _, cat_dict in coco.cats.items()
                }
            if self._class_name_to_class_id == {}:
                self._class_name_to_class_id = {
                    cat_dict['name']: cat_dict['id']
                    for _, cat_dict in coco.cats.items()
                }

            self._classes = sorted(self._class_name_to_class_id.keys())
            self._annotation[split_name] = coco

            if self._only_mappings:
                return

            for image_id, annotation in tqdm(coco.imgToAnns.items()):
                image_path = os.path.join(
                    self._download_path, '{}{}'.format(split_name,
                                                       CocoParser._YEAR),
                    coco.imgs[image_id]['file_name'])
                boxes = []
                classes = []

                for obj in annotation:
                    if self._skip_crowd and obj['iscrowd']:
                        self._crowd_instances[split_name] += 1
                        continue
                    boxes.append(obj['bbox'])
                    classes.append(obj['category_id'])

                if len(classes) == 0:
                    self._skipped_samples[split_name] += 1
                    continue

                sample = {
                    'image': image_path,
                    'image_id': image_id,
                    'label': {
                        'boxes': _convert_box_format(boxes),
                        'classes': classes
                    }
                }
                self._data[split_name].append(sample)

        if not self._only_val:
            _build(self._train_annotations_path, 'train')
        _build(self._val_annotations_path, 'val')

        for split_name in ['train', 'val']:
            logging.info(
                'Successfully parsed {} {} samples from {} dataset'.format(
                    len(self._data[split_name]), split_name, CocoParser._NAME))

            logging.info('Skipped {} {} empty samples'.format(
                self._skipped_samples[split_name], split_name))

            if self._skip_crowd:
                logging.info(
                    'Skipped {} crowd instances from {} samples'.format(
                        self._crowd_instances[split_name], split_name))

    @property
    def annotation(self):
        return self._annotation
