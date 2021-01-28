import tensorflow as tf


class PiecewiseConstantDecayWithLinearWarmup(
        tf.keras.optimizers.schedules.PiecewiseConstantDecay):
    def __init__(self, warmup_learning_rate, warmup_steps, boundaries, values,
                 **kwargs):
        super(PiecewiseConstantDecayWithLinearWarmup,
              self).__init__(boundaries=[x - 1 for x in boundaries],
                             values=values,
                             name='piecewise_constant_decay_with_linear_warmup',
                             **kwargs)

        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self._step_size = self.values[0] - self.warmup_learning_rate

    def __call__(self, step):
        learning_rate = tf.cond(
            pred=tf.less(step, self.warmup_steps),
            true_fn=lambda:
            (self.warmup_learning_rate + tf.cast(step, dtype=tf.float32) /
                self.warmup_steps * self._step_size),
            false_fn=lambda: (super(PiecewiseConstantDecayWithLinearWarmup,
                                    self).__call__(step)))
        return learning_rate

    def get_config(self):
        config = {
            "warmup_learning_rate": self.warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
        }
        base_config = super(PiecewiseConstantDecayWithLinearWarmup,
                            self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InverseDecay(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_rate, name='inverse_decay'):
        super(InverseDecay, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.name = name

    def __call__(self, step):
        learning_rate = tf.math.divide_no_nan(
            self.initial_learning_rate,
            (1.0 + self.decay_rate * tf.cast(step, dtype=tf.float32)))
        return learning_rate

    def get_config(self):
        config = {
            "name": self.name,
            "initial_learning_rate": self.initial_learning_rate,
            "decay_rate": self.decay_rate,
        }
        base_config = super(InverseDecay,
                            self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
