from copy import deepcopy
import json

import tensorflow as tf
from absl import logging

from retinanet.optimizers.piecewise_constant_decay_with_warmup import \
    PiecewiseConstantDecayWithLinearWarmup


def build_optimizer(params, precision):
    _params = deepcopy(params)
    lr_params = _params.pop('lr_params', None)
    learning_rate_fn = PiecewiseConstantDecayWithLinearWarmup(
        lr_params.warmup_learning_rate, lr_params.warmup_steps,
        lr_params.boundaries, lr_params.values)
    _params['learning_rate'] = learning_rate_fn

    config = {
        'class_name': _params['name'],
        'config': _params
    }

    optimizer = tf.optimizers.get(config)
    if precision == 'mixed_float16':
        logging.info(
            'Wrapping optimizer with `LossScaleOptimizer` for AMP training')
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            optimizer, dynamic=True)

    logging.info(
        'Optimizer Config:\n{}'
        .format(json.dumps(optimizer.get_config(), indent=4)))

    return optimizer
