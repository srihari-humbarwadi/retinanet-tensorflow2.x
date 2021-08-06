import logging
import tensorflow as tf

from retinanet.dataloader.anchor_generator import AnchorBoxGenerator
from retinanet.dataloader.utils import convert_to_corners


class FuseDetections(tf.keras.layers.Layer):

    def __init__(self, min_level, max_level, **kwargs):
        super(FuseDetections, self).__init__(**kwargs)

        self.min_level = min_level
        self.max_level = max_level

    def call(self, predictions):
        class_predictions = predictions['class-predictions']
        box_predictions = predictions['box-predictions']

        class_logits = []
        encoded_boxes = []

        box_predictions_shape = \
            box_predictions[str(self.min_level)].get_shape().as_list()
        class_predictions_shape = \
            class_predictions[str(self.min_level)].get_shape().as_list()

        anchors_at_each_location = box_predictions_shape[-1] // 4
        num_classes = class_predictions_shape[-1] // anchors_at_each_location
        batch_size = box_predictions_shape[0] or 1

        for level in range(self.min_level, self.max_level + 1):
            level = str(level)
            _cls_preds = class_predictions[level]
            _box_preds = box_predictions[level]

            shape = _box_preds.get_shape().as_list()
            anchors_at_this_level = anchors_at_each_location * shape[1] * shape[2]

            class_logits += [
                tf.reshape(
                    _cls_preds,
                    shape=[batch_size, anchors_at_this_level, num_classes])
            ]
            encoded_boxes += [
                tf.reshape(
                    _box_preds,
                    shape=[batch_size, anchors_at_this_level, 4])
            ]

        class_logits = tf.concat(class_logits, axis=1)
        encoded_boxes = tf.concat(encoded_boxes, axis=1)

        return {
            'class_logits': class_logits,
            'encoded_boxes': encoded_boxes
        }


class TransformBoxesAndScores(tf.keras.layers.Layer):

    def __init__(self, params, **kwargs):

        super(TransformBoxesAndScores, self).__init__(**kwargs)

        self._input_shape = tf.tile(
            tf.expand_dims(tf.constant(
                params.input.input_shape,
                dtype=tf.float32), axis=0),
            multiples=[1, 2])

        self._anchors = AnchorBoxGenerator(
            *params.input.input_shape,
            params.architecture.feature_fusion.min_level,
            params.architecture.feature_fusion.max_level,
            params.anchor_params)

        self._box_variance = tf.convert_to_tensor(
            params.encoder_params.box_variance, dtype=tf.float32)

        self._scale_box_predictions = \
            params.encoder_params.scale_box_targets

    def _transform_box_predictions(self, box_predictions):
        boxes = box_predictions
        anchor_boxes = tf.expand_dims(self._anchors.boxes, axis=0)

        if self._scale_box_predictions:
            boxes = boxes * self._box_variance

        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] +
                anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        boxes_transformed = boxes_transformed / self._input_shape
        return tf.clip_by_value(boxes_transformed, 0.0, 1.0)

    def call(self, predictions):
        class_logits = predictions['class_logits']
        encoded_boxes = predictions['encoded_boxes']

        class_logits = tf.cast(class_logits, dtype=tf.float32)
        boxes = tf.cast(encoded_boxes, dtype=tf.float32)

        scores = tf.nn.sigmoid(class_logits)
        boxes = self._transform_box_predictions(boxes)

        return {'scores': scores, 'boxes': boxes}


class FilterTopKDetections(tf.keras.layers.Layer):

    def __init__(self, top_k=100, filter_per_class=True, **kwargs):
        super(FilterTopKDetections, self).__init__(**kwargs)

        self.top_k = top_k
        self.filter_per_class = filter_per_class

    def _filter_per_class(self, scores, boxes):
        _, num_anchors, num_classes = scores.get_shape().as_list()
        top_k = min(self.top_k, num_anchors)

        scores = tf.transpose(scores, [0, 2, 1])
        scores = tf.reshape(scores, [-1, num_anchors])

        scores, indices = tf.nn.top_k(
            scores,
            top_k,
            sorted=False)

        scores = tf.reshape(scores, [-1, num_classes, top_k])
        indices = tf.reshape(indices, [-1, num_classes, top_k])
        scores = tf.transpose(scores, [0, 2, 1])
        indices = tf.transpose(indices, [0, 2, 1])

        boxes = tf.gather(boxes, indices, batch_dims=1)

        return scores, boxes

    def _filter_global(self, scores, boxes):
        _, num_anchors, num_classes = scores.get_shape().as_list()
        top_k = min(self.top_k, num_anchors * num_classes)

        scores_reshaped = tf.reshape(scores,
                                     [-1, num_anchors * num_classes])
        _, indices = tf.math.top_k(scores_reshaped, top_k, sorted=False)
        anchor_indices = tf.expand_dims(indices // num_classes, 2)

        scores = tf.gather_nd(scores, anchor_indices, batch_dims=1)
        boxes = tf.gather_nd(boxes, anchor_indices, batch_dims=1)

        return scores, boxes

    def call(self, predictions):
        scores = predictions['scores']
        boxes = predictions['boxes']

        if self.filter_per_class:
            filtered_scores, filtered_boxes = self._filter_per_class(scores, boxes)

        else:
            filtered_scores, filtered_boxes = self._filter_global(scores, boxes)

        return {'scores': filtered_scores, 'boxes': filtered_boxes}


class GenerateDetections(tf.keras.layers.Layer):
    _SUPPORTED_NMS_MODES = [
        'CombinedNMS',
        'GlobalSoftNMS',
        'GlobalHardNMS',
        'PerClassSoftNMS',
        'PerClassHardNMS',
    ]

    def __init__(self,
                 iou_threshold=0.5,
                 score_threshold=0.05,
                 max_detections=100,
                 soft_nms_sigma=None,
                 num_classes=None,
                 mode='CombinedNMS',
                 **kwargs):

        if mode not in GenerateDetections._SUPPORTED_NMS_MODES:
            raise AssertionError(
                'Requested unsupported mode: {}, available modes are: {}'
                .format(mode, GenerateDetections._SUPPORTED_NMS_MODES))

        self._running_on_tpu = isinstance(
            tf.distribute.get_strategy(), tf.distribute.TPUStrategy)

        if self._running_on_tpu:
            if mode != 'GlobalHardNMS' and mode != 'PerClassHardNMS':
                raise AssertionError(
                    'Requested mode not supported on Cloud TPUs.'
                    ' Please use `GlobalHardNMS` or `PerClassHardNMS`')

            logging.warn('Running on NMS op on TPU')

        super(GenerateDetections, self).__init__(**kwargs)

        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.soft_nms_sigma = soft_nms_sigma
        self.num_classes = num_classes
        self.mode = mode

    def _combined_nms(self, predictions):

        scores = predictions['scores']
        boxes = predictions['boxes']

        if len(boxes.get_shape().as_list()) == 3:
            boxes = tf.expand_dims(boxes, axis=2)

        detections = tf.image.combined_non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_output_size_per_class=self.max_detections,
            max_total_size=self.max_detections,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            clip_boxes=False,
            name='combined_nms')

        return {
            'scores': detections.nmsed_scores,
            'boxes': detections.nmsed_boxes,
            'classes': detections.nmsed_classes,
            'valid_detections': detections.valid_detections,
        }

    def _global_nms(self, predictions, sigma):

        def _global_nms_single_image(boxes, scores):

            selected_indices, selected_scores, valid_detections = \
                tf.raw_ops.NonMaxSuppressionV5(
                    boxes=boxes,
                    scores=tf.reduce_max(scores, axis=-1),
                    max_output_size=self.max_detections,
                    iou_threshold=1.0 if not sigma else self.iou_threshold,
                    score_threshold=self.score_threshold,
                    soft_nms_sigma=sigma / 2,
                    pad_to_max_output_size=True)

            selected_boxes = tf.gather(boxes, selected_indices)
            selected_classes = tf.gather(tf.argmax(scores, axis=-1),
                                         selected_indices)

            selected_scores = tf.where(
                tf.less(tf.range(self.max_detections), valid_detections),
                selected_scores, -1.0)

            selected_classes = tf.where(
                tf.less(tf.range(self.max_detections), valid_detections),
                selected_classes, -1)

            return (selected_scores, selected_boxes, selected_classes,
                    valid_detections)

        scores = predictions['scores']
        boxes = predictions['boxes']

        detections = tf.vectorized_map(
            fn=lambda x: _global_nms_single_image(x[0], x[1]),
            elems=(boxes, scores))

        return {
            'scores': detections[0],
            'boxes': detections[1],
            'classes': detections[2],
            'valid_detections': detections[3]
        }

    def _tpu_global_hard_nms(self, predictions):
        scores = tf.reduce_max(predictions['scores'], axis=-1)
        classes = tf.argmax(predictions['scores'], axis=-1)
        boxes = predictions['boxes']

        batch_size, num_anchors, _ = boxes.get_shape().as_list()

        if batch_size is None:
            batch_size = tf.shape(batch_size)[0]

        indices, valid_detections = \
            tf.image.non_max_suppression_padded(
                boxes=boxes,
                scores=scores,
                max_output_size=self.max_detections,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
                canonicalized_coordinates=True,
                pad_to_max_output_size=True)

        idx = tf.expand_dims(tf.range(self.max_detections), axis=0)
        idx = tf.tile(idx, [batch_size, 1])

        valid_detection_mask = tf.less(idx, tf.expand_dims(valid_detections,
                                                           axis=-1))
        valid_detection_mask = tf.reshape(valid_detection_mask,
                                          [batch_size, self.max_detections, 1])

        scores = tf.expand_dims(scores, axis=-1)
        classes = tf.expand_dims(tf.cast(classes, dtype=tf.float32), axis=-1)

        boxes_classes_scores = tf.concat([boxes, classes, scores], axis=-1)
        selected_boxes_classes_scores = tf.gather(boxes_classes_scores,
                                                  indices,
                                                  batch_dims=1)

        selected_boxes_classes_scores = tf.where(
            valid_detection_mask, selected_boxes_classes_scores,
            tf.ones_like(selected_boxes_classes_scores) * -1.0)

        selected_scores = selected_boxes_classes_scores[:, :, 5]
        selected_boxes = selected_boxes_classes_scores[:, :, :4]
        selected_classes = tf.cast(selected_boxes_classes_scores[:, :, 4],
                                   dtype=tf.int32)

        return {
            'scores': selected_scores,
            'boxes': selected_boxes,
            'classes': selected_classes,
            'valid_detections': valid_detections
        }

    def _per_class_nms(self, predictions, sigma):

        def _per_class_nms_single_image(_boxes, _scores):

            def _nms(class_id):
                boxes_for_class_id = \
                    _boxes[:, min(num_boxes_per_class - 1, class_id), :]
                scores_for_class_id = _scores[:, class_id]

                selected_indices, selected_scores, _ = \
                    tf.raw_ops.NonMaxSuppressionV5(
                        boxes=boxes_for_class_id,
                        scores=scores_for_class_id,
                        max_output_size=self.max_detections,
                        iou_threshold=1.0 if not sigma else self.iou_threshold,
                        score_threshold=self.score_threshold,
                        soft_nms_sigma=sigma / 2,
                        pad_to_max_output_size=True)

                selected_boxes = tf.gather(boxes_for_class_id, selected_indices)

                return (
                    selected_scores,
                    selected_boxes,
                )

            filtered_scores = []
            filtered_boxes = []
            filtered_classes = []

            for class_id in range(self.num_classes):
                nms_results = _nms(class_id)
                filtered_scores.append(nms_results[0])
                filtered_boxes.append(nms_results[1])
                filtered_classes.append(tf.fill([self.max_detections],
                                                class_id))

            filtered_scores = tf.concat(filtered_scores, axis=0)
            filtered_boxes = tf.concat(filtered_boxes, axis=0)
            filtered_classes = tf.concat(filtered_classes, axis=0)

            filtered_scores, topk_indices = tf.nn.top_k(filtered_scores,
                                                        k=self.max_detections,
                                                        sorted=False)

            filtered_boxes = tf.gather(filtered_boxes, topk_indices)
            filtered_classes = tf.gather(filtered_classes, topk_indices)
            _valid_detections = tf.reduce_sum(
                tf.cast(tf.greater(filtered_scores, 0), tf.int32))

            filtered_scores = tf.where(
                tf.less(tf.range(self.max_detections), _valid_detections),
                filtered_scores, -1.0)

            filtered_classes = tf.where(
                tf.less(tf.range(self.max_detections), _valid_detections),
                filtered_classes, -1)

            return (
                filtered_scores,
                filtered_boxes,
                filtered_classes,
                _valid_detections
            )

        scores = predictions['scores']
        boxes = predictions['boxes']

        boxes_shape = boxes.get_shape().as_list()
        batch_size = boxes_shape[0] or 1

        if len(boxes_shape) == 3:
            boxes = tf.expand_dims(boxes, axis=2)
            num_boxes_per_class = 1

        else:
            num_boxes_per_class = boxes_shape[2]

        nmsed_scores = []
        nmsed_boxes = []
        nmsed_classes = []
        valid_detections = []

        for i in range(batch_size):
            detections = _per_class_nms_single_image(boxes[i], scores[i])
            nmsed_scores.append(detections[0])
            nmsed_boxes.append(detections[1])
            nmsed_classes.append(detections[2])
            valid_detections.append(detections[3])

        nmsed_scores = tf.stack(nmsed_scores, axis=0)
        nmsed_boxes = tf.stack(nmsed_boxes, axis=0)
        nmsed_classes = tf.stack(nmsed_classes, axis=0)
        valid_detections = tf.stack(valid_detections, axis=0)

        return {
            'scores': nmsed_scores,
            'boxes': nmsed_boxes,
            'classes': nmsed_classes,
            'valid_detections': valid_detections
        }

    def call(self, predictions):

        predictions = tf.nest.map_structure(
            lambda x: tf.cast(x, dtype=tf.float32), predictions)

        if self.mode == 'CombinedNMS':
            return self._combined_nms(predictions)

        if self.mode == 'GlobalSoftNMS':
            return self._global_nms(predictions, sigma=self.soft_nms_sigma)

        if self.mode == 'GlobalHardNMS':
            if self._running_on_tpu:
                return self._tpu_global_hard_nms(predictions)

            return self._global_nms(predictions, sigma=0.0)

        if self.mode == 'PerClassSoftNMS':
            return self._per_class_nms(predictions, sigma=self.soft_nms_sigma)

        if self.mode == 'PerClassHardNMS':
            return self._per_class_nms(predictions, sigma=0.0)
