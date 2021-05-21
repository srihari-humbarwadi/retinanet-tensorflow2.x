import tensorflow as tf
from retinanet.model.layers import NearestUpsampling2D
from retinanet.model.utils import get_normalization_op


class MultiLevelAttentionFusion(tf.keras.layers.Layer):

    def __init__(self,
                 filters=256,
                 projection_dim=64,
                 min_level=3,
                 max_level=7,
                 backbone_max_level=5,
                 conv_2d_op_params=None,
                 normalization_op_params=None,
                 use_lateral_conv=True,
                 **kwargs):
        super(MultiLevelAttentionFusion, self).__init__(**kwargs)

        self.filters = filters
        self.min_level = min_level
        self.max_level = max_level
        self.backbone_max_level = backbone_max_level
        self.use_lateral_conv = use_lateral_conv

        self.projection_convs = {}
        self.attention_convs = {}

        self.intermediate_norms = {}
        self.projection_norms = {}
        self.output_convs = {}
        self.output_norms = {}

        self.num_features = (self.backbone_max_level - self.min_level + 1)
        normalization_op = get_normalization_op(**normalization_op_params)

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

        if self.use_lateral_conv:
            self.lateral_convs = {}

        for level in range(min_level, backbone_max_level + 1):
            level = str(level)

            if self.use_lateral_conv:
                self.lateral_convs[level] = conv_2d_op(filters=self.filters,
                                                       kernel_size=1,
                                                       strides=1,
                                                       padding='same',
                                                       use_bias=conv_2d_op_params.use_bias_before_bn,
                                                       name='l' + str(level) +
                                                       '-conv2d',
                                                       **kernel_initializer_config)

            self.projection_convs[level] = conv_2d_op(
                filters=projection_dim,
                kernel_size=1,
                use_bias=conv_2d_op_params.use_bias_before_bn,
                name='l' + str(level) + '-projection-conv2d')

            self.attention_convs[level] = conv_2d_op(
                filters=self.num_features *
                filters,
                kernel_size=1,
                name='l' + str(level) + '-attention-conv2d')

            self.intermediate_norms[level] = normalization_op(
                name='l' + str(level) + '-intermediate-batch_normalization')

            self.projection_norms[level] = normalization_op(
                name='l' + str(level) + '-projection-batch_normalization')

        if not max_level == backbone_max_level:
            
            for level in range(min_level, max_level + 1):
                level = str(level)
                self.output_norms[level] = normalization_op(name='p' + str(level) +
                                                            '-batch_normalization')
                self.output_convs[level] = conv_2d_op(
                    filters=self.filters,
                    kernel_size=3,
                    padding='same',
                    strides=2 if int(level) > backbone_max_level else 1,
                    use_bias=conv_2d_op_params.use_bias_before_bn,
                    name='p' + str(level) + '-conv2d',
                    **kernel_initializer_config)

    def call(self, features, training=False):
        intermediate_features = {}
        outputs = {}

        for level in range(self.min_level, self.backbone_max_level + 1):
            level = str(level)
            x = features[level]

            if self.use_lateral_conv:
                x = self.lateral_convs[level](x)

            x = self.intermediate_norms[level](x, training=training)
            x = tf.nn.relu(x)

            intermediate_features[level] = x

        for current_level in range(self.min_level, self.backbone_max_level + 1):

            fused_features = []
            for level in range(self.min_level, self.backbone_max_level + 1):
                x = intermediate_features[str(level)]

                if level > current_level:
                    x = NearestUpsampling2D(
                        scale=2**(level - current_level))(x)

                elif level < current_level:
                    x = tf.keras.layers.MaxPool2D(pool_size=2**(current_level -
                                                                level))(x)

                fused_features.append(x)

            fused_features = tf.stack(fused_features)

            x = tf.reduce_sum(fused_features, axis=0)
            x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
            x = self.projection_convs[str(current_level)](x)
            x = self.projection_norms[str(current_level)](x, training=training)
            x = tf.nn.relu(x)

            x = self.attention_convs[str(current_level)](x)
            x = tf.stack(
                tf.split(x, num_or_size_splits=self.num_features, axis=3))
            x = tf.nn.softmax(x, axis=0)

            fused_features = tf.reduce_sum(fused_features * x, axis=0)
            outputs[str(current_level)] = fused_features

        if not self.max_level == self.backbone_max_level:

            for level in range(self.min_level, self.max_level + 1):
                level = str(level)
                if int(level) <= self.backbone_max_level:
                    outputs[level] = self.output_convs[level](outputs[level])

                elif int(level) == self.backbone_max_level + 1:
                    outputs[level] = self.output_convs[level](
                        outputs[str(int(level) - 1)])

                else:
                    prev_level_output = \
                        tf.nn.relu(outputs[str(int(level) - 1)])

                    outputs[level] = self.output_convs[level](
                        prev_level_output)

            for level in range(self.min_level, self.max_level + 1):
                level = str(level)
                outputs[level] = self.output_norms[level](
                    outputs[level], training=training)

        return outputs
