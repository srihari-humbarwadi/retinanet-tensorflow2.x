import os

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

    if params.type == 'tpu':
        from cloud_tpu_client import Client

        logging.info('Creating TPU strategy')

        tpu_name = params.name

        if tpu_name == '':
            if 'TPU_NAME' not in os.environ:
                raise AssertionError(
                    'Failed to fetch TPU name, please set ENV VAR `TPU_NAME` or specify TPU name in config ')  # noqa: E501

            tpu_name = os.environ['TPU_NAME']
            logging.warning(
                'Using {} as TPU name from ENV VAR `TPU_NAME`'.format(tpu_name))

        else:
            if 'TPU_NAME' in os.environ:
                tpu_name = os.environ['TPU_NAME']

                logging.warning(
                    'Changed TPU name from {} to {} (overided with ENV VAR `TPU_NAME`)'  # noqa: E501
                    .format(params.name, tpu_name))

        if tpu_name not in {'', 'local'} and 'pod' not in tpu_name:
            # skip configuring TPU when TPU VM is detected
            logging.info('Configuring TPU: {} with correct tensorflow version'
                         .format(tpu_name))

            c = Client(tpu_name)
            c.configure_tpu_version(tf.__version__, restart_type='always')

            logging.info('Done Configuring TPU: {} with tensorflow version: {}'
                         .format(tpu_name, tf.__version__))

        resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
            tpu_name)

        return tf.distribute.TPUStrategy(resolver)

    raise ValueError('Unsupported strategy requested')
