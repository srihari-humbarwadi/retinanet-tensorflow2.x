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
    tf.keras.mixed_precision.set_policy(policy)

    logging.info('Compute dtype: {}'.format(policy.compute_dtype))
    logging.info('Variable dtype: {}'.format(policy.variable_dtype))
