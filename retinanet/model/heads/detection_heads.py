import tensorflow as tf


class DetectionHead(tf.keras.layers.Layer):

    def __init__(self,
                 num_head_convs=4,
                 filters=256,
                 output_filters=36,
                 prediction_bias_initializer='zeros',
                 **kwargs):
        super(DetectionHead, self).__init__(**kwargs)

        self.num_head_convs = num_head_convs
        self.filters = filters
        self.output_filters = filters
        self.prediction_bias_initializer = prediction_bias_initializer

        conv_2d_op = tf.keras.layers.Conv2D
        bn_op = tf.keras.layers.BatchNormalization

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

            norms = []
            for j in range(3, 8):
                norms += [
                    bn_op(momentum=0.997,
                          epsilon=1e-4,
                          name='{}-{}-p{}-bn'.format(self.name, i, j))
                ]

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

        for j, feature in enumerate(features):
            x = feature
            for i in range(self.num_head_convs):
                x = self.head_convs[i](x)
                x = self.head_norms[i][j](x, training=training)
                x = tf.nn.relu(x)

            x = self.prediction_conv(x)
            outputs['p{}-{}-predictions'.format(j + 3, self.name)] = x

        return outputs
