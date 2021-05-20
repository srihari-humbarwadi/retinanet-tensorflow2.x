import tensorflow as tf


class NearestUpsampling2D(tf.keras.layers.Layer):

    def __init__(self, scale, use_native=True, **kwargs):
        super(NearestUpsampling2D, self).__init__(**kwargs)
        self.scale = scale
        self.use_native = use_native

        if use_native:
            self._native_upscaling_op = tf.keras.layers.UpSampling2D(
                size=scale, interpolation='nearest', name=self.name)

    def call(self, images):
        if self.use_native:
            return self._native_upscaling_op(images)

        scale = self.scale
        size = tf.shape(images)
        batch_size, height, width, channels = size[0], size[1], size[2], size[3]

        images = tf.stack([images] * self.scale, axis=3)
        images = tf.stack([images] * self.scale, axis=2)
        return tf.reshape(
            images, shape=[batch_size, height * scale, width * scale, channels])

    def get_config(self):
        config = {'scale': self.scale}
        base_config = super(NearestUpsampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
