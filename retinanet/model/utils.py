import functools

import tensorflow as tf
from absl import logging

from retinanet.model.layers.tpu_batch_normalization import TpuBatchNormalization


def get_normalization_op(**params):
    strategy = tf.distribute.get_strategy()
    num_replicas = strategy.num_replicas_in_sync

    if params.get('use_sync', None):
        if isinstance(strategy, tf.distribute.TPUStrategy):
            op = TpuBatchNormalization

        elif num_replicas > 1:
            op = tf.keras.layers.experimental.SyncBatchNormalization

    else:
        op = tf.keras.layers.BatchNormalization

    logging.debug('Using {}'.format(op.__class__.__name__))

    normalization_op = functools.partial(
        op,
        momentum=params.get('momentum'),
        epsilon=params.get('epsilon')
    )
    return normalization_op
