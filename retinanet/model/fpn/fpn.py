import functools

import tensorflow as tf

from retinanet.model.layers.nearest_upsampling import NearestUpsampling2D
from retinanet.model.utils import get_normalization_op


class FPN(tf.keras.Model):

    def __init__(self,
                 filters=256,
                 min_level=3,
                 max_level=7,
                 backbone_max_level=5,
                 **kwargs):
        super(FPN, self).__init__(**kwargs)

        self.filters = filters
        self.min_level = min_level
        self.max_level = max_level
        self.backbone_max_level = backbone_max_level

        conv_2d_op = tf.keras.layers.Conv2D
        normalization_op = get_normalization_op()

        self.upsample_op = functools.partial(NearestUpsampling2D, scale=2)
        self.lateral_convs = {}
        self.output_convs = {}
        self.output_norms = {}
        self.relu_ops = {}

        for level in range(min_level, backbone_max_level + 1):
            level = str(level)
            self.lateral_convs[level] = conv_2d_op(
                filters=self.filters,
                kernel_size=1,
                strides=1,
                kernel_initializer=tf.initializers.VarianceScaling(),
                padding='same',
                name='l' + str(level) + '-conv2d')

        for level in range(min_level, max_level + 1):
            level = str(level)
            self.output_norms[level] = normalization_op(
                name='p' + str(level) + '-batch_normalization')

            self.output_convs[level] = conv_2d_op(
                filters=self.filters,
                kernel_size=3,
                kernel_initializer=tf.initializers.VarianceScaling(),
                padding='same',
                strides=1 if int(level) < min_level + 3 else 2,
                name='p' + str(level) + '-conv2d')

        for level in range(backbone_max_level + 1, max_level):
            level = str(level)
            name = 'p{}-relu'.format(level)
            self.relu_ops[level] = tf.keras.layers.ReLU(name=name)

    def call(self, features, training=None):
        outputs = {}

        for level in range(self.min_level, self.backbone_max_level + 1):
            level = str(level)
            conv_layer = self.lateral_convs[level]
            outputs[level] = conv_layer(features[level])

        for level in range(self.backbone_max_level, self.min_level, -1):
            level = str(level)
            name = 'm{}-upsample'.format(int(level) - 1)
            outputs[str(int(level) - 1)] += self.upsample_op(name=name)(
                outputs[level])

        for level in range(self.min_level, self.max_level + 1):
            level = str(level)
            if int(level) <= self.backbone_max_level:
                outputs[level] = self.output_convs[level](outputs[level])

            elif int(level) == self.backbone_max_level + 1:
                outputs[level] = self.output_convs[level](
                    outputs[str(int(level) - 1)])

            else:
                prev_level_output = \
                    self.relu_ops[str(int(level) - 1)
                                  ](outputs[str(int(level) - 1)])

                outputs[level] = self.output_convs[level](prev_level_output)

        for level in range(self.min_level, self.max_level + 1):
            level = str(level)
            outputs[level] = self.output_norms[level](
                outputs[level], training=training)

        return outputs
