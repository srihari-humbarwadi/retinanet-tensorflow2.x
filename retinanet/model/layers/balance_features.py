import tensorflow as tf

from retinanet.model.layers.nearest_upsampling import NearestUpsampling2D


class BalanceFeatures(tf.keras.layers.Layer):

    def __init__(self, min_level, max_level, intermediate_level, **kwargs):

        if intermediate_level < min_level or intermediate_level > max_level:
            raise AssertionError('Invalide intermedite level passed')

        super(BalanceFeatures, self).__init__(**kwargs)

        self.min_level = min_level
        self.max_level = max_level
        self.intermediate_level = intermediate_level

    def call(self, features):
        resized_features = {}
        outputs = {}

        for level in range(self.min_level, self.max_level + 1):
            if level > self.intermediate_level:
                resized_features[str(level)] = NearestUpsampling2D(
                    scale=2**(level - self.intermediate_level),
                    name='p-{}-upsample'.format(level))(
                        features[str(level)])

            elif level < self.intermediate_level:
                resized_features[str(level)] = tf.keras.layers.MaxPool2D(
                    pool_size=2**(self.intermediate_level - level),
                    name='p-{}-downsample'.format(level))(
                        features[str(level)])

            else:
                resized_features[str(level)] = features[str(level)]

        averaged_feature = tf.add_n([x for x in resized_features.values()
                                     ]) / (self.max_level - self.min_level + 1)

        for level in range(self.min_level, self.max_level + 1):
            if level > self.intermediate_level:
                resized_averaged_feature = tf.keras.layers.MaxPool2D(
                    pool_size=2**(level - self.intermediate_level),
                    name='p-{}-downsample'.format(level))(averaged_feature)

            elif level < self.intermediate_level:
                resized_averaged_feature = NearestUpsampling2D(
                    scale=2**(self.intermediate_level - level),
                    name='p-{}-upsample'.format(level))(averaged_feature)

            else:
                resized_averaged_feature = averaged_feature

            outputs[str(level)] = tf.keras.layers.Add(
                name='p-{}-average-add'.format(level))(
                [features[str(level)], resized_averaged_feature])

        return outputs
