from copy import deepcopy
import json

import tensorflow as tf
from absl import logging

from retinanet.optimizers.piecewise_constant_decay_with_warmup import \
    PiecewiseConstantDecayWithLinearWarmup
from retinanet.optimizers.cosine_decay_with_warmup import \
    CosineDecayWithLinearWarmup


def get_learning_rate_schedule(total_steps, params):
    _params = deepcopy(params)
    schedule_type = _params.pop('schedule_type', None)

    if schedule_type == 'piecewise_constant_decay':
        return PiecewiseConstantDecayWithLinearWarmup(**_params)

    if schedule_type == 'cosine_decay':
        _params['total_steps'] = total_steps
        return CosineDecayWithLinearWarmup(**_params)

    raise ValueError('Invalid learning rate schedule requested')


def build_optimizer(params, train_steps, precision):
    # workaround for bug in easydict `pop` method
    _params = dict(deepcopy(params))

    lr_params = _params.pop('lr_params', None)
    use_moving_average = _params.pop('use_moving_average', None)
    moving_average_decay = _params.pop('moving_average_decay', None)

    _ = _params.pop('global_clipnorm', None)
    _ = _params.pop('clipnorm', None)

    _params['learning_rate'] = get_learning_rate_schedule(train_steps, lr_params)

    config = {
        'class_name': _params['name'],
        'config': _params
    }

    optimizer = tf.optimizers.get(config)

    if use_moving_average:
        try:
            import tensorflow_addons as tfa

            optimizer = tfa.optimizers.MovingAverage(
                optimizer=optimizer,
                average_decay=moving_average_decay,
                dynamic_decay=True)

        except ImportError:
            logging.warning(
                'Failed to import TensorFlow Addons, building optimizer with '
                '`use_moving_average=False`')

    if precision == 'mixed_float16':
        logging.info(
            'Wrapping optimizer with `LossScaleOptimizer` for AMP training')
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            optimizer, dynamic=True)

    logging.info(
        'Optimizer Config:\n{}'
        .format(json.dumps(optimizer.get_config(), indent=4)))

    return optimizer
