import functools

import tensorflow as tf

from retinanet.model.layers.feature_fusion import FeatureFusion
from retinanet.model.layers.nearest_upsampling import NearestUpsampling2D
from retinanet.model.utils import Identity, get_normalization_op


class GenericFPN(tf.keras.layers.Layer):

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

        super(GenericFPN, self).__init__(**kwargs)

        self.filters = filters
        self.min_level = min_level
        self.max_level = max_level
        self.fusion_mode = fusion_mode
        self.backbone_max_level = backbone_max_level

        normalization_op = get_normalization_op(**normalization_op_params)

        self.upsample_op = functools.partial(NearestUpsampling2D, scale=2)
        self.downsample_op = functools.partial(
            tf.keras.layers.MaxPool2D, pool_size=2)

        self.channel_normalize_convs = {}
        self.channel_normalize_norms = {}
        self.output_convs = {}
        self.output_norms = {}
        self.output_activations = {}
        self.fusion_ops = {}
        self.fusion_activation_ops = {}

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
            self.channel_normalize_convs[level] = conv_2d_op(
                filters=self.filters,
                kernel_size=1,
                strides=1,
                padding='same',
                name='p{}-in-channel-normalize-conv-1x1'.format(level),
                **kernel_initializer_config)
            self.channel_normalize_norms[level] = normalization_op(
                name='p{}-in-channel-normalize-batch_normalization'.format(level))

        for level in range(min_level, max_level + 1):
            level = str(level)

            self.output_convs[level] = conv_2d_op(
                filters=self.filters,
                kernel_size=3,
                padding='same',
                strides=1,
                use_bias=conv_2d_op_params.use_bias_before_bn,
                name='p{}-out-conv-3x3'.format(level),
                **kernel_initializer_config)

            self.output_norms[level] = normalization_op(
                name='p{}-out-batch_normalization'.format(level))

            if int(level) != min_level:
                self.fusion_ops[level] = FeatureFusion(
                    mode=fusion_mode,
                    filters=filters,
                    name='p{}-in-fusion-with-p{}-in-upsampled'.format(
                        str(int(level) - 1), level))
                self.fusion_activation_ops[level] = activation_fn(
                    name='p{}-in-fusion-with-p{}-in-upsampled'.format(
                        str(int(level) - 1), level))

        # conv_1x1 for backbone_max_level to normalize num_channels
        # this will be used to generate the first coarse level. In
        # the original FPN this should be added on top of C5, which will
        # then be used to generate C6
        self._backbone_max_level_conv_1x1 = conv_2d_op(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding='same',
            name='backbone_max_level_conv_1x1',
            **kernel_initializer_config)
        self._backbone_max_level_bn = normalization_op(
            name='backbone_max_level_batch_normalization')

    def call(self, features, training=None):
        outputs = features

        # name the features for better visualization
        for level in range(self.min_level, self.backbone_max_level + 1):
            x = outputs[str(level)]
            outputs[str(level)] = Identity(name='p{}-in'.format(level))(x)

        # generate additional coarse feature maps from backbone_max_level
        for level in range(self.backbone_max_level + 1, self.max_level + 1):
            x = outputs[str(level - 1)]
            if level == self.backbone_max_level + 1:
                x = self._backbone_max_level_conv_1x1(x)
                x = self._backbone_max_level_bn(x, training=training)
            x = self.downsample_op(name='p{}-in-downsample'.format(level - 1))(x)
            outputs[str(level)] = Identity(name='p{}-in'.format(level))(x)

        for level in range(self.min_level, self.backbone_max_level + 1):
            level = str(level)
            conv_layer = self.channel_normalize_convs[level]
            norm_layer = self.channel_normalize_norms[level]
            x = conv_layer(outputs[level])
            outputs[level] = norm_layer(x, training=training)

        # add top down pathway
        for level in range(self.max_level, self.min_level, -1):
            level = str(level)
            name = 'p{}-in-upsampled'.format(level)
            x = self.fusion_ops[level]([outputs[str(int(level) - 1)],
                                        self.upsample_op(name=name)(outputs[level])])
            outputs[str(int(level) - 1)] = self.fusion_activation_ops[level](x)

        # add output convs
        for level in range(self.min_level, self.max_level + 1):
            level = str(level)
            x = self.output_convs[level](outputs[level])
            x = self.output_norms[level](x, training=training)
            outputs[str(level)] = Identity(name='p{}-out'.format(level))(x)

        return outputs
