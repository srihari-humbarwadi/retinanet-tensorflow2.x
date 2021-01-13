import tensorflow as tf


class InverseDecay(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_rate):
        super(InverseDecay, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def __call__(self, step):
        learning_rate = tf.math.divide_no_nan(
            self.initial_learning_rate,
            (1.0 + self.decay_rate * tf.cast(step, dtype=tf.float32)))
        return learning_rate

    def get_config(self):
        config = {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_rate": self.decay_rate,
        }
        base_config = super(InverseDecay,
                            self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
