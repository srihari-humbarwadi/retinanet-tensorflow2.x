import json
import os
from copy import deepcopy

from absl import logging
from tqdm import tqdm


class COCOConverter:

    _COCO_DICT = {
        'info': {
            'discription': '',
            'url': '',
            'version': '',
            'year': '',
            'contributor': '',
            'date_created': ''
        },
        'licenses': [
            {
                'url': '',
                'id': 1,
                'name': ''
            }
        ],
        'categories': [],
        'images': [],
        'annotations': []
    }

    _CATEGORY_DICT = {
        'supercategory': 1,
        'id': 1,
        'name': ''
    }

    _IMAGE_DICT = {
        'id': None,
        'license': 1,
        'coco_url': '',
        'flickr_url': '',
        'width': None,
        'height': None,
        'filename': '',
        'date_captures': ''
    }

    _ANNOTATION_DICT = {
        'id': None,
        'image_id': None,
        'iscrowd': 0,
        'category_id': None,
        'segmentation': [1],
        'area': None,
        'bbox': [None, None, None, None]
    }

    def __init__(
            self,
            parsed_dataset_json,
            label_map,
            output_dir='./dataset',
            only_val=True):

        self.parsed_dataset = COCOConverter._read_json(parsed_dataset_json)
        self.label_map = COCOConverter._read_json(label_map)
        self.output_dir = output_dir
        self._only_val = only_val

    @staticmethod
    def _read_json(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def convert(self):
        logging.info('Populating category information for {} categories'
                     .format(len(self.label_map)))

        categories = []
        for class_id, class_name in self.label_map.items():
            category_dict = deepcopy(COCOConverter._CATEGORY_DICT)

            category_dict['supercategory'] = int(class_id)
            category_dict['id'] = int(class_id)
            category_dict['name'] = class_name
            categories.append(category_dict)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        def _convert_annotations(split):
            logging.info('Convert {} split from {} dataset'
                         .format(split, self.parsed_dataset['name']))

            converted_json = deepcopy(COCOConverter._COCO_DICT)
            converted_json['categories'] = categories

            convert_json_path = os.path.join(
                self.output_dir,
                'instances_{}.json'.format(split))

            for sample in tqdm(self.parsed_dataset['dataset'][split]):
                image_dict = deepcopy(COCOConverter._IMAGE_DICT)

                image_dict['id'] = int(sample['image_id'])
                image_dict['width'] = sample['image_width']
                image_dict['height'] = sample['image_height']
                image_dict['filename'] = os.path.basename(sample['image'])

                converted_json['images'].append(image_dict)

                labels = sample['label']
                for idx, (box, class_id) in enumerate(zip(labels['boxes'], labels['classes'])):  # noqa: E501
                    annotation_dict = deepcopy(COCOConverter._ANNOTATION_DICT)

                    annotation_dict['id'] = len(converted_json['annotations'])
                    annotation_dict['image_id'] = int(sample['image_id'])
                    annotation_dict['category_id'] = class_id

                    x1, y1, x2, y2 = box
                    _coco_format_box = [x1, y1, x2 - x1, y2 - y1]

                    annotation_dict['area'] = float((x2 - x1) * (y2 - y1))
                    annotation_dict['bbox'] = list(map(float, _coco_format_box))

                    converted_json['annotations'].append(annotation_dict)

            logging.info(
                'Successfully converted {} samples containing {} annotations from {} split'  # noqa: E501
                .format(
                    len(self.parsed_dataset['dataset'][split]),
                    len(converted_json['annotations']),
                    split))

            with open(convert_json_path, 'w') as f:
                logging.info('Dumping {} split as json to {}'
                             .format(split, convert_json_path))
                json.dump(converted_json, f, indent=4)

        if not self._only_val:
            _convert_annotations('train')
        _convert_annotations('val')
