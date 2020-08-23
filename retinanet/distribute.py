import tensorflow as tf
from absl import logging


def get_strategy(params):
    if params.type == 'gpu':
        logging.info('Creating GPU strategy')
        return tf.distribute.OneDeviceStrategy(device='/gpu:0')

    if params.type == 'cpu':
        logging.info('Creating CPU strategy')
        return tf.distribute.OneDeviceStrategy(device='/cpu:0')

    if params.type == 'multi_gpu':
        logging.info('Creating Multi GPU strategy')
        return tf.distribute.MirroredStrategy()

    elif params.type == 'tpu':
        logging.info('Creating TPU strategy')
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
            params.name)
        return tf.distribute.TPUStrategy(resolver)
    raise ValueError('Unsupported strategy requested')
