from absl import logging
import tensorflow as tf
import tensorflow_addons as tfa


class ImageDistortion:
    "Applies random image distortions"

    _SUPPORTED_OPS = ['brightness', 'contrast', 'grayscale', 'grayscale_blend']
    _MAX_DELTA = 10.0

    def __init__(self, ops_list=None, num_layers=1, delta=5.0, apply_prob=0.5):
        if ops_list is None:
            ops_list = ImageDistortion._SUPPORTED_OPS
            logging.warn('`ops_list` not defined, using all available ops')
        else:
            for op in ops_list:
                if op not in ImageDistortion._SUPPORTED_OPS:
                    raise AssertionError('Requested unsupported operation: {}. '
                                         'Supported operations: {}'.format(
                                             op,
                                             ImageDistortion._SUPPORTED_OPS))
        logging.info('Using {} for image distortions'
                     .format(ImageDistortion._SUPPORTED_OPS))

        self._op_list = ops_list
        self._num_layers = num_layers
        self._delta = delta
        self._apply_prob = apply_prob

    def _brightness(self, image):
        max_delta = self._delta / ImageDistortion._MAX_DELTA
        return tf.image.random_brightness(image, max_delta)

    def _contrast(self, image):
        delta = self._delta / ImageDistortion._MAX_DELTA
        return tf.image.random_contrast(image, 1.0 - delta, 1.0 + 1e-4)

    def _grayscale(self, image):
        grayscale_image = tf.image.rgb_to_grayscale(image)
        return tf.image.grayscale_to_rgb(grayscale_image)

    def _grayscale_blend(self, image):
        factor = self._delta / ImageDistortion._MAX_DELTA
        grayscale_image = self._grayscale(image)
        return tfa.image.blend(image, grayscale_image, factor)

    def _distort(self, image):
        image = tf.cast(image, dtype=tf.uint8)

        for _ in range(self._num_layers):
            ops = []
            for idx, op_name in enumerate(self._op_list):
                ops.append((idx, lambda image=image: tf.cast(
                    getattr(self, '_' + op_name)(image), dtype=tf.uint8)))

            op_idx = tf.random.uniform([], 0, len(ops), dtype=tf.int32)
            image = tf.switch_case(branch_index=op_idx, branch_fns=ops)

        return image

    def distort(self, image):
        return tf.cond(
            pred=tf.random.uniform([], 0, 1) < self._apply_prob,
            true_fn=lambda: tf.cast(self._distort(image), dtype=tf.float32),
            false_fn=lambda: tf.identity(image))
