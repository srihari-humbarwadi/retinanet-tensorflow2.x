import numpy as np
import tensorflow as tf

from retinanet.model.head.detection_head import DetectionHead


def build_detection_heads(
        params,
        min_level,
        max_level,
        conv_2d_op_params=None,
        normalization_op_params=None,
        activation_fn=None):

    if activation_fn is None:
        raise ValueError('`activation_fn` cannot be None')

    box_head = DetectionHead(
        num_convs=params.num_convs,
        filters=params.filters,
        output_filters=params.num_anchors * 4,
        min_level=min_level,
        max_level=max_level,
        prediction_bias_initializer='zeros',
        conv_2d_op_params=conv_2d_op_params,
        normalization_op_params=normalization_op_params,
        activation_fn=activation_fn,
        name='box-head')

    prior_prob_init = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
    class_head = DetectionHead(
        num_convs=params.num_convs,
        filters=params.filters,
        output_filters=params.num_anchors*params.num_classes,
        min_level=min_level,
        max_level=max_level,
        prediction_bias_initializer=prior_prob_init,
        conv_2d_op_params=conv_2d_op_params,
        normalization_op_params=normalization_op_params,
        activation_fn=activation_fn,
        name='class-head')

    return box_head, class_head


def build_auxillary_head(
        num_convs,
        filters,
        num_anchors,
        min_level,
        max_level,
        conv_2d_op_params=None,
        normalization_op_params=None,
        activation_fn=None):

    if activation_fn is None:
        raise ValueError('`activation_fn` cannot be None')

    prior_prob_init = tf.constant_initializer(-np.log((1 - 0.5) / 0.5))
    auxillary_head = DetectionHead(
        num_convs=num_convs,
        filters=filters,
        output_filters=num_anchors,
        min_level=min_level,
        max_level=max_level,
        prediction_bias_initializer=prior_prob_init,
        conv_2d_op_params=conv_2d_op_params,
        normalization_op_params=normalization_op_params,
        activation_fn=activation_fn,
        name='auxillary-head')

    return auxillary_head
