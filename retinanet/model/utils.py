import functools

import tensorflow as tf
from absl import logging

from retinanet.model.layers.tpu_batch_normalization import TpuBatchNormalization


def get_normalization_op(**params):
    strategy = tf.distribute.get_strategy()
    num_replicas = strategy.num_replicas_in_sync

    if params.get('use_sync', None):
        if isinstance(strategy, tf.distribute.TPUStrategy):
            logging.debug('Using `{}`'.format('TpuBatchNormalization'))
            op = TpuBatchNormalization

        elif num_replicas > 1:
            logging.debug('Using `{}`'.format('SyncBatchNormalization'))
            op = tf.keras.layers.experimental.SyncBatchNormalization

        else:
            op = tf.keras.layers.BatchNormalization

    else:
        op = tf.keras.layers.BatchNormalization

    normalization_op = functools.partial(
        op,
        momentum=params.get('momentum'),
        epsilon=params.get('epsilon')
    )
    return normalization_op
