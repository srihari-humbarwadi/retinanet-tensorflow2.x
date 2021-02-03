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
            prediction_file_path):

        self._input_shape = input_shape
        self.annotation_file_path = annotation_file_path
        self.prediction_file_path = prediction_file_path

        self._coco_eval_obj = COCO(annotation_file_path)

        self._processed_detections = []

        logging.info('Initialized COCOEvaluator with {}'.format(
            self.annotation_file_path))

    def accumulate_results(self, results):
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
            resize_scale = resize_scales[i] / self._input_shape
            resize_scale = tf.tile(tf.expand_dims(
                resize_scale, axis=0), multiples=[1, 2])

            valid_detections = detections['valid_detections'][i].numpy()
            boxes = \
                detections['boxes'][i][:valid_detections].numpy(
                ) / resize_scale

            classes = detections['classes'][i][:valid_detections].numpy()
            scores = detections['scores'][i][:valid_detections].numpy()

            boxes = np.int32(boxes)
            boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]
            for box, int_id, score in zip(boxes, classes, scores):
                temp_dict = coco_eval_dict.copy()
                temp_dict['image_id'] = int(image_ids[i])
                temp_dict['category_id'] = int(int_id)
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
