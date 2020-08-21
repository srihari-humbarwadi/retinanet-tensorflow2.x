from absl import logging
import tensorflow as tf

STRATEGIES = {
    'gpu': tf.distribute.OneDeviceStrategy(device='/gpu:0'),
    'cpu': tf.distribute.OneDeviceStrategy(device='/cpu:0'),
    'multi_gpu': tf.distribute.MirroredStrategy()
}


def get_strategy(params):
    if params.type in ['gpu', 'cpu', 'multi_gpu']:
        logging.info('Creating {} strategy'.format(' '.join(
            [x.upper() for x in params.type.split('_')])))
        return STRATEGIES[params.type]

    elif params.type == 'tpu':
        logging.info('Creating TPU strategy')
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
            params.type.name)
        return tf.distribute.TPUStrategy(resolver)
    raise ValueError('Unsupported strategy requested')
