import tensorflow as tf

from retinanet.dataloader.anchor_generator import AnchorBoxGenerator
from retinanet.dataloader.preprocessing_pipeline import PreprocessingPipeline
from retinanet.dataloader.utils import compute_iou


class LabelEncoder:
    def __init__(self, params):
        self.input_shape = params.input.input_shape
        self.encoder_params = params.encoder_params
        self.anchors = AnchorBoxGenerator(*self.input_shape,
                                          params.architecture.fpn.min_level,
                                          params.architecture.fpn.max_level,
                                          params.anchor_params)
        self.preprocessing_pipeline = PreprocessingPipeline(
            self.input_shape, params.dataloader_params)

        self._all_unmatched = -1 * tf.ones(
            [self.anchors.boxes.get_shape().as_list()[0]], dtype=tf.int32)

    def _match_anchor_boxes(self, anchor_boxes, gt_boxes):
        if tf.shape(gt_boxes)[0] == 0:
            return self._all_unmatched

        iou_matrix = compute_iou(gt_boxes, anchor_boxes)

        max_ious = tf.reduce_max(iou_matrix, axis=0)
        matched_gt_idx = tf.argmax(iou_matrix, axis=0, output_type=tf.int32)

        matches = tf.where(tf.greater(max_ious, self.encoder_params.match_iou),
                           matched_gt_idx, -1)
        matches = tf.where(
            tf.logical_and(
                tf.greater_equal(max_ious, self.encoder_params.ignore_iou),
                tf.greater(self.encoder_params.match_iou, max_ious)), -2,
            matches)

        best_matched_anchors = tf.argmax(iou_matrix,
                                         axis=-1,
                                         output_type=tf.int32)
        best_matched_anchors_one_hot = tf.one_hot(
            best_matched_anchors, depth=tf.shape(iou_matrix)[-1])
        matched_anchors = tf.reduce_max(best_matched_anchors_one_hot, axis=0)
        matched_anchors_gt_idx = tf.argmax(best_matched_anchors_one_hot,
                                           axis=0,
                                           output_type=tf.int32)
        matches = tf.where(tf.cast(matched_anchors, dtype=tf.bool),
                           matched_anchors_gt_idx, matches)
        return matches

    def _compute_box_target(self, matched_gt_boxes, matches, eps=1e-8):
        matched_gt_boxes = tf.maximum(matched_gt_boxes, eps)
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - self.anchors.boxes[:, :2]) /
                self.anchors.boxes[:, 2:],
                tf.math.log(
                    matched_gt_boxes[:, 2:] / self.anchors.boxes[:, 2:]),
            ],
            axis=-1,
        )
        positive_mask = tf.expand_dims(tf.greater_equal(matches, 0), axis=-1)
        positive_mask = tf.broadcast_to(positive_mask, tf.shape(box_target))

        box_target = tf.where(positive_mask, box_target, 0.0)

        if self.encoder_params.scale_box_targets:
            box_target = box_target / tf.convert_to_tensor(
                self.encoder_params.box_variance, dtype=tf.float32)
        return box_target

    @staticmethod
    def _pad_labels(gt_boxes, cls_ids):
        gt_boxes = tf.concat([tf.stack([tf.zeros(4), tf.zeros(4)]), gt_boxes],
                             axis=0)
        cls_ids = tf.concat([
            tf.squeeze(tf.stack([-2 * tf.ones(1), -1 * tf.ones(1)])), cls_ids
        ],
            axis=0)
        return gt_boxes, cls_ids

    def encode_sample(self, sample):
        image, gt_boxes, cls_ids = self.preprocessing_pipeline(sample)
        matches = self._match_anchor_boxes(self.anchors.boxes, gt_boxes)
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        gt_boxes, cls_ids = LabelEncoder._pad_labels(gt_boxes, cls_ids)
        gt_boxes = tf.gather(gt_boxes, matches + 2)
        cls_target = tf.gather(cls_ids, matches + 2)
        box_target = self._compute_box_target(gt_boxes, matches)

        boundaries = self.anchors.anchor_boundaries
        targets = {'class-targets': {}, 'box-targets': {}}

        for i in range(5):
            fh = tf.math.ceil(self.input_shape[0] / (2**(i + 3)))
            fw = tf.math.ceil(self.input_shape[1] / (2**(i + 3)))
            targets['class-targets'][i + 3] = tf.reshape(
                cls_target[boundaries[i]:boundaries[i + 1]],
                shape=[fh, fw, self.anchors._num_anchors])
            targets['box-targets'][i + 3] = tf.reshape(
                box_target[boundaries[i]:boundaries[i + 1]],
                shape=[fh, fw, 4 * self.anchors._num_anchors])

        num_positives = tf.reduce_sum(
            tf.cast(tf.greater(matches, -1), dtype=tf.float32))
        targets['num-positives'] = num_positives
        return image, targets
