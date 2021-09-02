import functools

import tensorflow as tf

from retinanet.model.layers.feature_fusion import FeatureFusion
from retinanet.model.layers.nearest_upsampling import NearestUpsampling2D
from retinanet.model.utils import get_normalization_op


class FPN(tf.keras.layers.Layer):

    def __init__(self,
                 filters=256,
                 min_level=3,
                 max_level=7,
                 backbone_max_level=5,
                 fusion_mode=None,
                 conv_2d_op_params=None,
                 normalization_op_params=None,
                 activation_fn=None,
                 **kwargs):

        if activation_fn is None:
            raise ValueError('`activation_fn` cannot be None')

        super(FPN, self).__init__(**kwargs)

        self.filters = filters
        self.min_level = min_level
        self.max_level = max_level
        self.fusion_mode = fusion_mode
        self.backbone_max_level = backbone_max_level

        normalization_op = get_normalization_op(**normalization_op_params)

        self.upsample_op = functools.partial(NearestUpsampling2D, scale=2)
        self.lateral_convs = {}
        self.output_convs = {}
        self.output_norms = {}
        self.fusion_ops = {}
        self.activation_ops = {}

        if not conv_2d_op_params.use_seperable_conv:
            conv_2d_op = tf.keras.layers.Conv2D
            kernel_initializer_config = {
                'kernel_initializer': tf.initializers.VarianceScaling()
            }

        else:
            conv_2d_op = tf.keras.layers.SeparableConv2D
            kernel_initializer_config = {
                'depthwise_initializer': tf.initializers.VarianceScaling(),
                'pointwise_initializer': tf.initializers.VarianceScaling()
            }

        for level in range(min_level, backbone_max_level + 1):
            level = str(level)
            self.lateral_convs[level] = conv_2d_op(
                filters=self.filters,
                kernel_size=1,
                strides=1,
                padding='same',
                name='l' + str(level) + '-conv2d',
                **kernel_initializer_config)

            if int(level) != min_level:
                self.fusion_ops[level] = FeatureFusion(
                    mode=fusion_mode,
                    filters=filters,
                    name='fusion-l' + str(int(level) - 1) + '-m' + level)

        for level in range(min_level, max_level + 1):
            level = str(level)
            self.output_norms[level] = normalization_op(
                name='p' + str(level) + '-batch_normalization')

            self.output_convs[level] = conv_2d_op(
                filters=self.filters,
                kernel_size=3,
                padding='same',
                strides=2 if int(level) > backbone_max_level else 1,
                use_bias=conv_2d_op_params.use_bias_before_bn,
                name='p' + str(level) + '-conv2d',
                **kernel_initializer_config)

        for level in range(backbone_max_level + 1, max_level):
            level = str(level)
            self.activation_ops[level] = activation_fn(name='p{}'.format(level))

    def call(self, features, training=None):
        outputs = {}

        for level in range(self.min_level, self.backbone_max_level + 1):
            level = str(level)
            conv_layer = self.lateral_convs[level]
            outputs[level] = conv_layer(features[level])

        for level in range(self.backbone_max_level, self.min_level, -1):
            level = str(level)
            name = 'm{}-upsample'.format(level)
            outputs[str(int(level) - 1)] = \
                self.fusion_ops[level]([outputs[str(int(level) - 1)],
                                        self.upsample_op(name=name)(outputs[level])])

        for level in range(self.min_level, self.max_level + 1):
            level = str(level)
            if int(level) <= self.backbone_max_level:
                outputs[level] = self.output_convs[level](outputs[level])

            elif int(level) == self.backbone_max_level + 1:
                outputs[level] = self.output_convs[level](
                    outputs[str(int(level) - 1)])

            else:
                prev_level_output = \
                    self.activation_ops[str(int(level) - 1)
                                        ](outputs[str(int(level) - 1)])

                outputs[level] = self.output_convs[level](prev_level_output)

        for level in range(self.min_level, self.max_level + 1):
            level = str(level)
            outputs[level] = self.output_norms[level](
                outputs[level], training=training)

        return outputs
