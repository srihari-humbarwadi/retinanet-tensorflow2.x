import tensorflow as tf

from retinanet.model.utils import get_normalization_op


class DetectionHead(tf.keras.Model):

    def __init__(self,
                 num_convs=4,
                 filters=256,
                 output_filters=36,
                 min_level=3,
                 max_level=7,
                 prediction_bias_initializer='zeros',
                 normalization_op_params=None,
                 **kwargs):
        super(DetectionHead, self).__init__(**kwargs)

        self.num_head_convs = num_convs
        self.filters = filters
        self.output_filters = filters
        self.min_level = min_level
        self.max_level = max_level
        self.prediction_bias_initializer = prediction_bias_initializer

        conv_2d_op = tf.keras.layers.Conv2D
        normalization_op = get_normalization_op(**normalization_op_params)

        self.head_convs = []
        self.head_norms = []
        self.relu_ops = []

        for i in range(num_convs):
            self.head_convs += [
                conv_2d_op(
                    filters=filters,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    name='{}-class-{}-conv2d'.format(self.name, i),
                    kernel_initializer=tf.keras.initializers.RandomNormal(
                        stddev=0.01),
                    bias_initializer=tf.zeros_initializer())
            ]

            norms = {}
            for level in range(min_level, max_level + 1):
                level = str(level)
                norms[level] = normalization_op(
                    name='{}-{}-p{}-batch_normalization'.format(self.name, i, level))

            self.head_norms += [norms]
            self.relu_ops += [
                tf.keras.layers.ReLU(
                    name='{}-class-{}-relu'.format(self.name, i))
            ]

        self.prediction_conv = conv_2d_op(
            filters=output_filters,
            kernel_size=3,
            strides=1,
            padding='same',
            name='{}-prediction-conv2d'.format(self.name),
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
            bias_initializer=self.prediction_bias_initializer,
            dtype=tf.float32)

    def call(self, features, training=None):
        outputs = {}

        for level in range(self.min_level, self.max_level + 1):
            level = str(level)
            x = features[level]

            for i in range(self.num_head_convs):
                x = self.head_convs[i](x)
                x = self.head_norms[i][level](x, training=training)
                x = self.relu_ops[i](x)

            outputs[level] = self.prediction_conv(x)

        return outputs
