import tensorflow as tf
from retinanet.model.builder import LOSS


@LOSS.register_module('retinanet')
class RetinaNetLoss:
    def __init__(self, num_classes, params):
        self.box_loss = BoxLoss(params.smooth_l1_loss)
        self.class_loss = ClassLoss(num_classes, params.focal_loss)

        self._box_loss_weight = tf.convert_to_tensor(params.box_loss_weight)
        self._class_loss_weight = tf.convert_to_tensor(
            params.class_loss_weight)

    def __call__(self, targets, predictions):
        normalizer = tf.reduce_sum(targets['num-positives']) + 1.0
        class_loss = self.class_loss(targets['class-targets'],
                                     predictions['class-predictions'],
                                     normalizer)
        box_loss = self.box_loss(targets['box-targets'],
                                 predictions['box-predictions'], normalizer)

        weighted_loss = self._box_loss_weight * box_loss + \
            self._class_loss_weight * class_loss

        return {
            'box-loss': box_loss,
            'class-loss': class_loss,
            'weighted-loss': weighted_loss,
            'num-anchors-matched': normalizer
        }


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
        for i in range(3, 8):
            loss.append(self.box_loss(targets[i], predictions[i], normalizer))
        return tf.math.add_n(loss)


class ClassLoss:
    def __init__(self, num_classes, params):
        self.class_loss = FocalLoss(params.alpha, params.gamma)
        self._num_classes = num_classes

    def __call__(self, targets, predictions, normalizer):
        loss = []
        for i in range(3, 8):
            n, h, w, c = targets[i].get_shape().as_list()
            per_level_loss = self.class_loss(
                tf.reshape(
                    tf.one_hot(tf.cast(targets[i], dtype=tf.int32),
                               depth=self._num_classes),
                    [n, h, w, c * self._num_classes]), predictions[i],
                normalizer)
            ignore_mask = tf.expand_dims(tf.where(tf.equal(targets[i], -2.0),
                                                  0.0, 1.0),
                                         axis=-1)
            ignore_mask = tf.tile(ignore_mask,
                                  multiples=[1, 1, 1, 1, self._num_classes])
            ignore_mask = tf.reshape(ignore_mask, tf.shape(per_level_loss))
            per_level_loss = per_level_loss * ignore_mask
            loss.append(tf.reduce_sum(per_level_loss))
        return tf.math.add_n(loss)
