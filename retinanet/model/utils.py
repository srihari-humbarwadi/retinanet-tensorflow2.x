import functools

import tensorflow as tf
from absl import logging


def get_normalization_op(**params):
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

    if params.get('use_sync', None) and num_replicas > 1:
        logging.debug('Using `{}`'.format('SyncBatchNormalization'))
        op = tf.keras.layers.experimental.SyncBatchNormalization

    else:
        op = tf.keras.layers.BatchNormalization

    normalization_op = functools.partial(
        op,
        momentum=params.get('momentum'),
        epsilon=params.get('epsilon')
    )
    return normalization_op


class GenericActivation(tf.keras.layers.Layer):
    def __init__(self, forward_fn, activation_type, name, **kwargs):
        super(GenericActivation, self).__init__(
            name=name + '-' + activation_type, **kwargs)

        self._forward_fn = forward_fn
        self._activation_type = activation_type

    def call(self, tensor):
        return self._forward_fn(tensor)


def get_activation_op(activation_type):
    _SUPPORTED_ACTIVATIONS = [
        'relu',
        'relu6',
        'swish',
    ]

    if activation_type not in _SUPPORTED_ACTIVATIONS:
        raise AssertionError('Unsupported activation requested. Available '
                             'activations: {}'.format(_SUPPORTED_ACTIVATIONS))

    logging.info('Setting activation type to: {}'.format(activation_type))

    def _fn(tensor):
        if activation_type == 'relu':
            return tf.nn.relu(tensor)

        elif activation_type == 'relu6':
            return tf.nn.relu6(tensor)

        elif activation_type == 'swish':
            return tf.nn.swish(tensor)

    activation_fn = functools.partial(
        GenericActivation, forward_fn=_fn, activation_type=activation_type)
    return activation_fn
