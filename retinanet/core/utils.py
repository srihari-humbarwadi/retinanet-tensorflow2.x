import os

import tensorflow as tf
from absl import logging

from retinanet.core.optimizer import PiecewiseConstantDecayWithLinearWarmup


def get_optimizer(params):
    lr_params = params.pop('lr_params', None)
    learning_rate_fn = PiecewiseConstantDecayWithLinearWarmup(
        lr_params.warmup_learning_rate, lr_params.warmup_steps,
        lr_params.boundaries, lr_params.values)
    params['learning_rate'] = learning_rate_fn

    config = {
        'class_name': params['name'],
        'config': params
    }
    optimizer = tf.optimizers.get(config)

    return optimizer


def add_l2_regularization(weight, alpha=0.0001):
    def _add_l2_regularization():
        return alpha * tf.nn.l2_loss(weight)

    return _add_l2_regularization


def get_normalization_op():
    use_sync_bn = tf.distribute.get_strategy().num_replicas_in_sync > 1
    use_sync_bn = use_sync_bn and 'USE_SYNC_BN' in os.environ

    if use_sync_bn:
        logging.debug('Using SyncBatchNormalization')
        return tf.keras.layers.experimental.SyncBatchNormalization

    return tf.keras.layers.BatchNormalization


def set_precision(precision):
    policy = tf.keras.mixed_precision.Policy(precision)
    tf.keras.mixed_precision.set_global_policy(policy)

    logging.info('Compute dtype: {}'.format(policy.compute_dtype))
    logging.info('Variable dtype: {}'.format(policy.variable_dtype))


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

        resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
            tpu_name)
        return tf.distribute.TPUStrategy(resolver)

    raise ValueError('Unsupported strategy requested')
