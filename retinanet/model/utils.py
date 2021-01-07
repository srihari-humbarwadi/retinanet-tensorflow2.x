import tensorflow as tf

from retinanet.lr_schedules import PiecewiseConstantDecayWithLinearWarmup


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
