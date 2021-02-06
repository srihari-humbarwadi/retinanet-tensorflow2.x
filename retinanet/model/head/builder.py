import numpy as np
import tensorflow as tf

from retinanet.model.head.detection_head import DetectionHead


def build_heads(params, min_level, max_level, normalization_op_params=None):
    box_head = DetectionHead(
        num_convs=params.num_convs,
        filters=params.filters,
        output_filters=params.num_anchors * 4,
        min_level=min_level,
        max_level=max_level,
        prediction_bias_initializer='zeros',
        normalization_op_params=normalization_op_params,
        name='box-head')

    prior_prob_init = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
    class_head = DetectionHead(
        num_convs=params.num_convs,
        filters=params.filters,
        output_filters=params.num_anchors*params.num_classes,
        min_level=min_level,
        max_level=max_level,
        prediction_bias_initializer=prior_prob_init,
        normalization_op_params=normalization_op_params,
        name='class-head')

    return box_head, class_head
