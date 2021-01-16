import tensorflow as tf

from retinanet.dataloader.anchor_generator import AnchorBoxGenerator
from retinanet.dataloader.utils import convert_to_corners


class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(self, params, **kwargs):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = params.architecture.num_classes
        self.confidence_threshold = params.inference.confidence_threshold
        self.nms_iou_threshold = params.inference.nms_iou_threshold
        self.max_detections_per_class = \
            params.inference.max_detections_per_class
        self.max_detections = params.inference.max_detections
        self.pre_nms_top_k = params.inference.pre_nms_top_k
        self._input_shape = tf.tile(
            tf.expand_dims(tf.constant(
                params.input.input_shape,
                dtype=tf.float32), axis=0),
            multiples=[1, 2])

        self._anchors = AnchorBoxGenerator(
            *params.input.input_shape,
            params.anchor_params)
        self._box_variance = tf.convert_to_tensor(
            params.encoder_params.box_variance,
            dtype=tf.float32)
        self._scale_box_predictions = \
            params.encoder_params.scale_box_targets

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions
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
        return boxes_transformed

    def _filter_top_k(self, class_predictions, boxes):
        class_predictions = tf.transpose(class_predictions, [0, 2, 1])
        class_predictions = tf.reshape(
            class_predictions, [-1, tf.shape(self._anchors.boxes)[0]])

        top_k_class_predictions, top_k_indices = tf.nn.top_k(
            class_predictions, self.pre_nms_top_k)
        top_k_class_predictions = tf.transpose(
            tf.reshape(top_k_class_predictions,
                       [-1, self.num_classes, self.pre_nms_top_k]), [0, 2, 1])
        top_k_indices = tf.transpose(
            tf.reshape(top_k_indices,
                       [-1, self.num_classes, self.pre_nms_top_k]), [0, 2, 1])

        top_k_boxes = tf.gather(boxes, top_k_indices, batch_dims=1)
        return top_k_class_predictions, top_k_boxes

    def call(self, predictions):
        box_predictions, class_predictions = predictions

        class_predictions = tf.cast(class_predictions, dtype=tf.float32)
        box_predictions = tf.cast(box_predictions, dtype=tf.float32)

        class_predictions = tf.nn.sigmoid(class_predictions)
        boxes = self._decode_box_predictions(self._anchors.boxes[None, ...],
                                             box_predictions)

        if self.pre_nms_top_k > 0:
            top_k_class_predictions, top_k_boxes = self._filter_top_k(
                class_predictions, boxes)

        else:
            top_k_boxes = tf.expand_dims(boxes, axis=2)
            top_k_class_predictions = class_predictions

        return tf.image.combined_non_max_suppression(
            top_k_boxes,
            top_k_class_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )
