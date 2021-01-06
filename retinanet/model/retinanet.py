import functools

import numpy as np
import tensorflow as tf

from retinanet.model.fpn import fpn_builder


def retinanet_builder(input_shape, params):

    if tf.distribute.get_strategy().num_replicas_in_sync > 1:
        use_sync = True
    else:
        use_sync = False

    fpn = fpn_builder(input_shape, params)

    k_init = tf.keras.initializers.RandomNormal(stddev=0.01)
    b_init = tf.zeros_initializer()
    prior_prob = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))

    conv_2d_op = tf.keras.layers.Conv2D

    normalization_op = tf.keras.layers.experimental.SyncBatchNormalization \
        if use_sync else tf.keras.layers.BatchNormalization

    bn_op = functools.partial(
        normalization_op,
        momentum=0.997,
        epsilon=1e-3 if use_sync else 1e-4)

    conv_3x3 = functools.partial(
        conv_2d_op,
        filters=256,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=k_init,
        bias_initializer=b_init)

    conv2d_same_pad = functools.partial(
        conv_2d_op,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5))

    class_convs = []
    box_convs = []

    for i in range(params.num_head_convs):
        class_convs += [conv_3x3(name='class-' + str(i))]
        box_convs += [conv_3x3(name='box-' + str(i))]

    box_convs += [
        conv2d_same_pad(filters=params.num_anchors * 4,
                        name='box-predictions',
                        bias_initializer=b_init,
                        dtype=tf.float32)
    ]
    class_convs += [
        conv2d_same_pad(filters=params.num_anchors * params.num_classes,
                        name='class-predictions',
                        bias_initializer=prior_prob,
                        dtype=tf.float32)
    ]

    box_bns = [
        bn_op(name='box-{}-{}'.format(i, j))
        for i in range(params.num_head_convs) for j in range(3, 8)
    ]
    class_bns = [
        bn_op(name='class-{}-{}'.format(i, j), )
        for i in range(params.num_head_convs) for j in range(3, 8)
    ]

    class_outputs = {}
    box_outputs = {}

    for i, output in enumerate(fpn.outputs):
        class_x = box_x = output
        for j in range(params.num_head_convs):
            class_x = class_convs[j](class_x)
            class_x = tf.nn.relu(class_bns[i + 5 * j](class_x))
            box_x = box_convs[j](box_x)
            box_x = tf.nn.relu(box_bns[i + 5 * j](box_x))
        class_outputs[i + 3] = class_convs[-1](class_x)
        box_outputs[i + 3] = box_convs[-1](box_x)

    outputs = {
        'class-predictions': class_outputs,
        'box-predictions': box_outputs
    }
    return tf.keras.Model(inputs=fpn.inputs, outputs=outputs, name='retinanet')
