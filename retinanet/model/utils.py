import os

import tensorflow as tf
from absl import logging


def get_normalization_op():
    use_sync_bn = tf.distribute.get_strategy().num_replicas_in_sync > 1
    use_sync_bn = use_sync_bn and 'USE_SYNC_BN' in os.environ

    if use_sync_bn:
        logging.debug('Using SyncBatchNormalization')
        return tf.keras.layers.experimental.SyncBatchNormalization

    return tf.keras.layers.BatchNormalization
