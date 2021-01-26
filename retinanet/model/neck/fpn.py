import functools

import tensorflow as tf

from retinanet.core.layers.nearest_upsampling import NearestUpsampling2D
from retinanet.core.utils import get_normalization_op
from retinanet.model.builder import NECK


@NECK.register_module('fpn_v1')
class FPN(tf.keras.Model):
    """ FPN builder class """

    def __init__(self, inputs, params):
        assert isinstance(inputs, (list, tuple)
                          ), f"list or tuple expected, passed {type(inputs)}"
        conv_2d_op = tf.keras.layers.Conv2D
        c3, c4, c5 = inputs

        normalization_op = get_normalization_op()
        bn_op = functools.partial(
            normalization_op,
            momentum=0.997,
            epsilon=1e-4)

        upsample_op = functools.partial(NearestUpsampling2D, scale=2)
        relu_op = functools.partial(tf.keras.layers.ReLU)

        min_level = 3
        max_level = 7

        lateral_convs = []
        output_convs = []
        output_bns = []

        conv2d_same_pad = functools.partial(
            conv_2d_op,
            filters=params.architecture.neck.filters,
            kernel_initializer=tf.initializers.VarianceScaling(),
            padding='same')

        for i in range(min_level, max_level + 1):
            if i < min_level + 3:
                lateral_convs += [
                    conv2d_same_pad(kernel_size=1, strides=1, name='l' + str(i))
                ]
            strides = 1 if i < min_level + 3 else 2
            output_convs += [
                conv2d_same_pad(kernel_size=3, strides=strides, name='p' + str(i))
            ]

            output_bns += [bn_op(name='p' + str(i) + '-bn')]

        l3, l4, l5 = [conv(x) for conv, x in zip(lateral_convs, (c3, c4, c5))]

        m4 = tf.keras.layers.Add(name='m4')([l4, upsample_op(name='m4_up')(l5)])
        m3 = tf.keras.layers.Add(name='m3')([l3, upsample_op(name='m3_up')(m4)])

        p3 = output_convs[0](m3)
        p4 = output_convs[1](m4)
        p5 = output_convs[2](l5)
        p6 = output_convs[3](p5)
        p7 = output_convs[4](relu_op(name='p6-relu')(p6))

        outputs = [bn(x) for bn, x in zip(output_bns, (p3, p4, p5, p6, p7))]
        super().__init__(inputs=inputs, outputs=outputs, name='fpn')
