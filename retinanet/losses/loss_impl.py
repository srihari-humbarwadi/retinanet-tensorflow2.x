import tensorflow as tf


class SmoothL1Loss:
    def __init__(self, delta):
        self._loss_fn = tf.losses.Huber(delta=delta, reduction='sum')

    def __call__(self, y_true, y_pred, normalizer):
        positive_mask = tf.not_equal(y_true, 0.0)
        loss = self._loss_fn(y_true[..., None],
                             y_pred[..., None],
                             sample_weight=positive_mask)
        loss /= 4.0 * normalizer
        return loss


class FocalLoss:
    def __init__(self, alpha, gamma):
        self._alpha = alpha
        self._gamma = gamma

    def __call__(self, y_true, y_pred, normalizer):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                                logits=y_pred)
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha,
                         (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1.0 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        loss /= normalizer
        return loss


class BoxLoss:
    def __init__(self, params):
        self.box_loss = SmoothL1Loss(params.delta)

    def __call__(self, targets, predictions, normalizer):
        loss = []
        for key in targets.keys():
            loss.append(self.box_loss(
                targets[key], predictions[key], normalizer))
        return tf.math.add_n(loss)


class ClassLoss:
    def __init__(self, num_classes, params):
        self.class_loss = FocalLoss(params.alpha, params.gamma)
        self._num_classes = num_classes

    def __call__(self, targets, predictions, normalizer):
        loss = []
        for key in targets.keys():
            n, h, w, c = targets[key].get_shape().as_list()
            per_level_loss = self.class_loss(
                tf.reshape(
                    tf.one_hot(tf.cast(targets[key], dtype=tf.int32),
                               depth=self._num_classes),
                    [n, h, w, c * self._num_classes]), predictions[key],
                normalizer)
            ignore_mask = tf.expand_dims(tf.where(tf.equal(targets[key], -2.0),
                                                  0.0, 1.0),
                                         axis=-1)
            ignore_mask = tf.tile(ignore_mask,
                                  multiples=[1, 1, 1, 1, self._num_classes])
            ignore_mask = tf.reshape(ignore_mask, tf.shape(per_level_loss))
            per_level_loss = per_level_loss * ignore_mask
            loss.append(tf.reduce_sum(per_level_loss))
        return tf.math.add_n(loss)
