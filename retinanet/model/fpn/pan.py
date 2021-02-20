import tensorflow as tf

from retinanet.model.fpn.fpn import FPN
from retinanet.model.layers.feature_fusion import FeatureFusion
from retinanet.model.utils import get_normalization_op


class PAN(tf.keras.Model):

    def __init__(self,
                 filters=256,
                 min_level=3,
                 max_level=7,
                 backbone_max_level=5,
                 fusion_mode=None,
                 conv_2d_op_params=None,
                 normalization_op_params=None,
                 **kwargs):

        super(PAN, self).__init__(**kwargs)

        self.fpn = FPN(
            filters=filters,
            min_level=min_level,
            max_level=max_level,
            backbone_max_level=backbone_max_level,
            fusion_mode=fusion_mode,
            use_residual_connections=False,
            conv_2d_op_params=conv_2d_op_params,
            normalization_op_params=normalization_op_params,
            name='fpn')

        self.filters = filters
        self.min_level = min_level
        self.max_level = max_level
        self.fusion_mode = fusion_mode

        normalization_op = get_normalization_op(**normalization_op_params)

        self.downsample_convs = {}
        self.output_convs = {}
        self.output_norms = {}
        self.fusion_ops = {}
        self.relu_ops = {}

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

        for level in range(min_level, max_level + 1):
            level = str(level)

            if int(level) > min_level:
                self.fusion_ops[level] = FeatureFusion(
                    mode=fusion_mode,
                    filters=filters,
                    name='fusion-p' + str(int(level) - 1) + '-p' + level)

            if int(level) < max_level:
                self.downsample_convs[level] = conv_2d_op(
                    filters=self.filters,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    name='p' + str(level) + '-downsample' + '-conv2d',
                    **kernel_initializer_config)

            self.output_convs[level] = conv_2d_op(
                filters=self.filters,
                kernel_size=3,
                padding='same',
                strides=1,
                use_bias=conv_2d_op_params.use_bias_before_bn,
                name='n' + str(level) + '-conv2d',
                **kernel_initializer_config)

            self.output_norms[level] = normalization_op(
                name='n' + str(level) + '-batch_normalization')

    def call(self, features, training=None):

        fpn_features = self.fpn(features, training=training)

        outputs = {
            str(self.min_level): fpn_features[str(self.min_level)]
        }

        for level in range(self.min_level + 1, self.max_level + 1):
            level = str(level)
            outputs[level] = self.fusion_ops[level]([
                fpn_features[level],
                self.downsample_convs[str(int(level) - 1)
                                      ](outputs[str(int(level) - 1)])
            ])

        for level in range(self.min_level, self.max_level + 1):
            level = str(level)
            x = self.output_convs[level](outputs[level])
            outputs[level] = self.output_norms[level](x, training=training)

        return outputs
