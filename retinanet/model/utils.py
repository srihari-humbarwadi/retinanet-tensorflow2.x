import functools

import tensorflow as tf
from absl import logging


def get_normalization_op(**params):
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

    if params.get('use_sync', None) and num_replicas > 1:
        logging.debug('Using SyncBatchNormalization')
        op = tf.keras.layers.experimental.SyncBatchNormalization

    else:
        op = tf.keras.layers.BatchNormalization

    normalization_op = functools.partial(
        op,
        momentum=params.get('momentum'),
        epsilon=params.get('epsilon')
    )
    return normalization_op
