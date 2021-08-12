import json

import numpy as np
import tensorflow as tf
from absl import logging
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCOEvaluator:
    def __init__(
            self,
            input_shape,
            annotation_file_path,
            prediction_file_path,
            remap_class_ids=False):

        self._input_shape = input_shape
        self.annotation_file_path = annotation_file_path
        self.prediction_file_path = prediction_file_path
        self._remap_class_ids = remap_class_ids

        self._coco_eval_obj = COCO(annotation_file_path)

        sorted_classes = sorted([
            category_info['name']
            for cat_id, category_info in self._coco_eval_obj.cats.items()])

        self._class_name_to_orig_class_id = {
            category_info['name']: category_info['id']
            for cat_id, category_info in self._coco_eval_obj.cats.items()
            }
        self._sorted_class_name_to_class_id = {
            class_name: idx for idx, class_name in enumerate(sorted_classes)
            }
        self._sorted_class_id_to_class_name = {
            idx: class_name for idx, class_name in enumerate(sorted_classes)
            }

        self._processed_detections = []

        logging.info('Initialized COCOEvaluator with {}'.format(
            self.annotation_file_path))

        if remap_class_ids:
            logging.warning('Evaluating with `remap_class_ids=True`')
        else:
            logging.warning('Evaluating with `remap_class_ids=False`')

    def _maybe_remap_class_ids(self, class_id):
        if self._remap_class_ids:
            class_name = self._sorted_class_id_to_class_name[class_id]
            return self._class_name_to_orig_class_id[class_name]
        return class_id

    def accumulate_results(self, results, rescale_detections=True):
        image_ids = results['image_id']
        detections = results['detections']
        resize_scales = results['resize_scale']

        batch_size = len(image_ids)

        coco_eval_dict = {
            'image_id': None,
            'category_id': None,
            'bbox': [],
            'score': None
        }

        for i in range(batch_size):

            valid_detections = detections['valid_detections'][i].numpy()
            boxes = detections['boxes'][i][:valid_detections].numpy()
            classes = detections['classes'][i][:valid_detections].numpy()
            scores = detections['scores'][i][:valid_detections].numpy()

            if rescale_detections:
                resize_scale = resize_scales[i] / self._input_shape
                resize_scale = tf.tile(tf.expand_dims(
                    resize_scale, axis=0), multiples=[1, 2])

                boxes /= resize_scale

            boxes = np.int32(boxes)
            boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]
            for box, int_id, score in zip(boxes, classes, scores):
                temp_dict = coco_eval_dict.copy()
                temp_dict['image_id'] = int(image_ids[i])
                temp_dict['category_id'] = self._maybe_remap_class_ids(
                    int(int_id))
                temp_dict['bbox'] = box.tolist()
                temp_dict['score'] = float(score)
                self._processed_detections += [temp_dict]

    def evaluate(self):
        logging.info('Dumping processed predictions to {}'.format(
            self.prediction_file_path))

        with open(self.prediction_file_path, 'w') as f:
            json.dump(self._processed_detections, f, indent=4)

        predictions = self._coco_eval_obj.loadRes(self.prediction_file_path)

        cocoEval = COCOeval(self._coco_eval_obj, predictions, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        scores = {
            'AP-IoU=0.50:0.95': cocoEval.stats[0],
            'AP-IoU=0.50': cocoEval.stats[1],
            'AP-IoU=0.75': cocoEval.stats[2],
            'AR-(all)-IoU=0.50:0.95': cocoEval.stats[6],
            'AR-(L)-IoU=0.50:0.95': cocoEval.stats[-1]
        }
        return scores

    @property
    def processed_detections(self):
        return self._processed_detections
