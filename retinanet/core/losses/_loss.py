import tensorflow as tf
import functools
from retinanet.model.builder import LOSS


@LOSS.register_module('retinanet')
class RetinaNetLoss:
    def __init__(self, num_classes, params):
        self.box_loss = BoxLoss(params.smooth_l1_loss)
        self.class_loss = ClassLoss(num_classes, params.focal_loss, reduction='sum')
        self.params = params
        self.num_classes = num_classes

        self._box_loss_weight = tf.convert_to_tensor(params.box_loss_weight)
        self._class_loss_weight = tf.convert_to_tensor(
            params.class_loss_weight)

    def __call__(self, targets, predictions):
        normalizer = tf.reduce_sum(targets['num-positives']) + 1.0
        class_targets, box_targets, class_preds, class_targets = self._concat_outs(targets, predictions)
        ignore_mask = tf.where(tf.equal(class_targets, -2.0), 0.0, 1.0)
        class_loss = self.class_loss(class_targets, class_preds, sample_weight=ignore_mask)
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

    def _concat_outs(self,targets, predictions ):
        class_targets = []
        box_targets = []
        box_preds = []
        class_preds = []
        min_level = 3 #self.params.architecture.neck.min_level
        max_level = 7 #self.params.architecture.neck.max_level
        for i in range(min_level, max_level + 1):
            key = 'p{}-predictions'.format(i)
            class_preds.append(tf.reshape(predictions['class-predictions'][key], (2,-1)))
            box_preds.append(tf.reshape(predictions['box-predictions'][key], (2, -1)))
            class_targets.append(tf.reshape(targets['class-targets'][i], (2, -1)))
            box_targets.append(tf.reshape(targets['box-targets'][i], (2, -1)))
        class_targets = tf.concat(class_targets, axis = -1)
        box_targets = tf.concat(box_targets, axis = -1)
        class_preds = tf.reshape(tf.concat(class_preds, axis = -1), (2, tf.shape(class_targets)[-1], self.num_classes))
        box_preds = tf.concat(box_preds, axis = -1)
        return class_targets, box_targets, class_preds, class_targets




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

    def __call__(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                                logits=y_pred)
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha,
                         (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1.0 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return loss


class BoxLoss:
    def __init__(self, params):
        self.box_loss = SmoothL1Loss(params.delta)

    def __call__(self, targets, predictions, normalizer):
        loss = []
        for i in range(3, 8):
            key = 'p{}-predictions'.format(i)
            loss.append(self.box_loss(targets[i], predictions[key], normalizer))
        return tf.math.add_n(loss)


class ClassLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, params, **kwargs):
        self.class_loss = FocalLoss(params.alpha, params.gamma)
        self._num_classes = num_classes
        super().__init__(**kwargs)

    def __call__(self, targets, predictions, sample_weight=None):
        loss = self.class_loss(
                tf.one_hot(tf.cast(targets, dtype=tf.int32),
                           depth=self._num_classes), predictions)
        return loss
