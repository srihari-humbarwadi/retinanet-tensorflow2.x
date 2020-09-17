import tensorflow as tf


def get_optimizer(name):
    if name == 'sgd':
        return tf.optimizers.SGD
    if name == 'adam':
        return tf.optimizers.Adam
    raise ValueError('Unsupported optimizer requested')


def add_l2_regularization(weight, alpha=0.0001):
    def _add_l2_regularization():
        return alpha * tf.nn.l2_loss(weight)

    return _add_l2_regularization
