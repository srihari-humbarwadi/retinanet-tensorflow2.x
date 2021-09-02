import functools

import tensorflow as tf

from retinanet.model.utils import Identity, get_normalization_op


class FPNBase(tf.keras.layers.Layer):
    def __init__(self,
                 filters=256,
                 min_level=3,
                 max_level=7,
                 backbone_max_level=5,
                 conv_2d_op_params=None,
                 normalization_op_params=None,
                 **kwargs):
        super(FPNBase, self).__init__(**kwargs)

        self.filters = filters
        self.min_level = min_level
        self.max_level = max_level
        self.backbone_max_level = backbone_max_level

        self._normalization_op = get_normalization_op(**normalization_op_params)
        self._downsample_op = functools.partial(
            tf.keras.layers.MaxPool2D, pool_size=2)

        if not conv_2d_op_params.use_seperable_conv:
            self._conv_2d_op = tf.keras.layers.Conv2D
            self._kernel_initializer_config = {
                'kernel_initializer': tf.initializers.VarianceScaling()
            }

        else:
            self._conv_2d_op = tf.keras.layers.SeparableConv2D
            self._kernel_initializer_config = {
                'depthwise_initializer': tf.initializers.VarianceScaling(),
                'pointwise_initializer': tf.initializers.VarianceScaling()
            }
        # conv_1x1 for backbone_max_level to normalize num_channels
        # this will be used to generate the first coarse level. In
        # the original FPN this should be added on top of C5, which will
        # then be used to generate C6
        self._backbone_max_level_conv_1x1 = self._conv_2d_op(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding='same',
            name='backbone_max_level_conv_1x1',
            **self._kernel_initializer_config)
        self._backbone_max_level_bn = self._normalization_op(
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
            x = self._downsample_op(name='p{}-in-downsample'.format(level - 1))(x)
            outputs[str(level)] = Identity(name='p{}-in'.format(level))(x)

        return outputs
