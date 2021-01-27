import tensorflow as tf

from retinanet.core.utils import get_normalization_op
from retinanet.model.builder import HEAD


@HEAD.register_module('retinanet_detection_head')
class DetectionHead(tf.keras.Model):

    def __init__(self,
                 num_head_convs=4,
                 filters=256,
                 output_filters=36,
                 min_level=3,
                 max_level=7,
                 prediction_bias_initializer='zeros',
                 **kwargs):
        super(DetectionHead, self).__init__(**kwargs)

        self.num_head_convs = num_head_convs
        self.filters = filters
        self.output_filters = filters
        self.min_level = min_level
        self.max_level = max_level
        self.prediction_bias_initializer = prediction_bias_initializer

        conv_2d_op = tf.keras.layers.Conv2D
        normalization_op = get_normalization_op()

        self.head_convs = []
        self.head_norms = []

        for i in range(num_head_convs):
            self.head_convs += [
                conv_2d_op(
                    filters=filters,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    name='{}-class-{}'.format(self.name, i),
                    kernel_initializer=tf.keras.initializers.RandomNormal(
                        stddev=0.01),
                    bias_initializer=tf.zeros_initializer())
            ]

            norms = {}
            for level in range(min_level, max_level + 1):
                level = str(level)
                norms[level] = normalization_op(
                    name='{}-{}-p{}-bn'.format(self.name, i, level))

            self.head_norms += [norms]

        self.prediction_conv = conv_2d_op(
            filters=output_filters,
            kernel_size=3,
            strides=1,
            padding='same',
            name='{}-prediction-conv'.format(self.name),
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
            bias_initializer=self.prediction_bias_initializer)

    def call(self, features, training=None):
        outputs = {}

        for level in range(self.min_level, self.max_level + 1):
            level = str(level)
            x = features[level]

            for i in range(self.num_head_convs):
                x = self.head_convs[i](x)
                x = self.head_norms[i][level](x, training=training)
                x = tf.nn.relu(x)

            x = self.prediction_conv(x)
            outputs['p{}-predictions'.format(level)] = x

        return outputs
