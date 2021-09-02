import tensorflow as tf


class FeatureFusion(tf.keras.layers.Layer):
    _SUPPORTED_FUSION_MODES = ['sum', 'fast_attention', 'fast_channel_attention']

    def __init__(self, mode, filters, name=None, **kwargs):

        if mode not in FeatureFusion._SUPPORTED_FUSION_MODES:
            raise AssertionError(
                'Requested unsupported mode: {}, available modes are: {}'
                .format(mode, FeatureFusion._SUPPORTED_FUSION_MODES))

        super(FeatureFusion, self).__init__(name=name, **kwargs)
        self.mode = mode
        self.filters = filters

    def build(self, input_shape):
        self.add_op = tf.keras.layers.Add(name='{}-add_op'.format(self.name))

        if not self.mode == 'sum':

            if self.mode == 'fast_attention':
                shape = [1]

            elif self.mode == 'fast_channel_attention':
                shape = [self.filters]

            self.lower_level_weight = self.add_weight(
                name='{}-lower-level-weight'.format(self.name),
                shape=shape,
                dtype=tf.float32,
                initializer=tf.initializers.Ones())

            self.upper_level_weight = self.add_weight(
                name='{}-upper-level-weight'.format(self.name),
                shape=shape,
                dtype=tf.float32,
                initializer=tf.initializers.Ones())

    def call(self, x):
        lower_level_feature, upper_level_feature = x

        if not self.mode == 'sum':
            lower_level_weight = tf.nn.relu(self.lower_level_weight)
            upper_level_weight = tf.nn.relu(self.upper_level_weight)

            weights_sum = lower_level_weight + upper_level_weight + 1e-4

            lower_level_feature = \
                lower_level_feature * lower_level_weight / weights_sum

            upper_level_feature = \
                upper_level_feature * upper_level_weight / weights_sum

        return self.add_op([lower_level_feature, upper_level_feature])
