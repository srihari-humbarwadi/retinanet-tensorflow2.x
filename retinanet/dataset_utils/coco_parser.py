import os

import numpy as np
from absl import logging
from pycocotools.coco import COCO
from tqdm import tqdm

from retinanet.dataset_utils.parser import Parser


class CocoParser(Parser):

    def __init__(self,
                 download_path,
                 remap_class_ids=False,
                 only_mappings=False,
                 only_val=False,
                 skip_crowd=True,
                 train_annotations_path='annotations/instances_train2017.json',
                 val_annotations_path='annotations/instances_val2017.json',
                 name='COCO',
                 year='2017'):
        super(CocoParser, self).__init__(
            download_path,
            name=name,
            remap_class_ids=remap_class_ids)

        self._year = year

        self._only_mappings = only_mappings
        self._only_val = only_val
        self._skip_crowd = skip_crowd

        if not only_val:
            self.train_annotations_path = os.path.join(
                download_path,
                train_annotations_path)

        self.val_annotations_path = os.path.join(
            download_path,
            val_annotations_path)

        self._crowd_instances = {'train': 0, 'val': 0}
        self._skipped_samples = {'train': 0, 'val': 0}
        self._skipped_annotations = {'train': 0, 'val': 0}
        self._annotation = {}
        self._build_dataset()

        if remap_class_ids:
            self._remap()

    def _build_dataset(self):

        def _is_box_valid(box, image_height, image_width):
            x, y, width, height = box

            if width <= 0 or height <= 0:
                return False

            if x + width > image_width or y + height > image_height:
                return False

            return True

        def _convert_box_format(boxes):
            boxes = np.array(boxes)
            return np.concatenate([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]],
                                  axis=-1)

        def _build(annotations_path, split_name):
            logging.info('Parsing {} split from {} dataset'.format(
                split_name, self._name))

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

            self._classes = set(self._class_name_to_class_id.keys())
            self._annotation[split_name] = coco

            if self._only_mappings:
                return

            for image_id, annotation in tqdm(coco.imgToAnns.items()):
                image_path = os.path.join(
                    self._download_path, '{}{}'.format(split_name,
                                                       self._year),
                    coco.imgs[image_id]['file_name'])
                boxes = []
                classes = []

                image_height = coco.imgs[image_id]['height']
                image_width = coco.imgs[image_id]['width']

                for obj in annotation:

                    if self._skip_crowd and obj['iscrowd']:
                        self._crowd_instances[split_name] += 1
                        continue

                    if not _is_box_valid(obj['bbox'],
                                         image_height,
                                         image_width):
                        self._skipped_annotations[split_name] += 1
                        continue

                    boxes.append(obj['bbox'])
                    classes.append(obj['category_id'])

                if len(classes) == 0:
                    self._skipped_samples[split_name] += 1
                    continue

                sample = {
                    'image': image_path,
                    'image_id': image_id,
                    'image_height': image_height,
                    'image_width': image_width,
                    'label': {
                        'boxes': _convert_box_format(boxes),
                        'classes': classes
                    }
                }
                self._data[split_name].append(sample)

        if not self._only_val:
            _build(self.train_annotations_path, 'train')
        _build(self.val_annotations_path, 'val')

        for split_name in ['train', 'val']:
            logging.info(
                'Successfully parsed {} {} samples from {} dataset'.format(
                    len(self._data[split_name]), split_name, self._name))

            logging.info('Skipped {} {} empty samples'.format(
                self._skipped_samples[split_name], split_name))

            logging.info('Skipped {} {} bad annotations'.format(
                self._skipped_annotations[split_name], split_name))

            if self._skip_crowd:
                logging.info(
                    'Skipped {} crowd instances from {} samples'.format(
                        self._crowd_instances[split_name], split_name))

    @ property
    def annotation(self):
        return self._annotation
