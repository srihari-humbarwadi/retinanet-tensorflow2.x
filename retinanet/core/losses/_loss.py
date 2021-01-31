import tensorflow as tf
import functools
from retinanet.model.builder import LOSS


@LOSS.register_module('retinanet')
class RetinaNetLoss:
    def __init__(self, params):
        self.box_loss = BoxLoss(params, reduction='sum')
        self.class_loss = ClassLoss(params, reduction='sum')
        self.params = params
        self.num_classes = params.architecture.num_classes

        self._box_loss_weight = tf.convert_to_tensor(params.loss.box_loss_weight)
        self._class_loss_weight = tf.convert_to_tensor(
            params.loss.class_loss_weight)

    def __call__(self, targets, predictions):
        class_targets, box_targets, class_preds, box_preds= self._concat_outs(targets, predictions)

        # ignore and positive masks
        ignore_mask = tf.where(tf.equal(class_targets, -2.0), 0.0, 1.0)
        positive_mask = tf.not_equal(box_targets, 0.0)

        class_loss = self.class_loss(class_targets, class_preds, sample_weight=ignore_mask)
        box_loss = self.box_loss(box_targets,
                                 box_preds, sample_weight=positive_mask)

        # reduce sum
        class_loss = tf.reduce_sum(class_loss)
        box_loss = tf.reduce_sum(box_loss)
        # normalize losses.
        normalizer = tf.reduce_sum(targets['num-positives']) + 1.0
        class_loss = class_loss / normalizer
        box_loss = box_loss / (4.0 * normalizer)

        weighted_loss = self._box_loss_weight * box_loss + \
            self._class_loss_weight * class_loss

        return {
            'box-loss': box_loss,
            'class-loss': class_loss,
            'weighted-loss': weighted_loss,
            'num-anchors-matched': normalizer
        }

    def _concat_outs(self, targets, predictions):
        class_targets = []
        box_targets = []
        box_preds = []
        class_preds = []
        min_level = self.params.architecture.neck.min_level
        max_level = self.params.architecture.neck.max_level
        b = self.params.training.batch_size.train
        for i in range(min_level, max_level + 1):
            key = 'p{}-predictions'.format(i)
            class_preds.append(tf.reshape(predictions['class-predictions'][key], (b,-1)))
            box_preds.append(tf.reshape(predictions['box-predictions'][key], (b, -1)))
            class_targets.append(tf.reshape(targets['class-targets'][i], (b, -1)))
            box_targets.append(tf.reshape(targets['box-targets'][i], (b, -1)))
        class_targets = tf.concat(class_targets, axis = -1)
        box_targets = tf.concat(box_targets, axis = -1)
        class_preds = tf.reshape(tf.concat(class_preds, axis = -1), (b, tf.shape(class_targets)[-1], self.num_classes))
        box_preds = tf.concat(box_preds, axis = -1)
        return class_targets, box_targets, class_preds, box_preds




class SmoothL1Loss:
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, y_true, y_pred):
        loss = tf.keras.losses.huber(y_true[..., None],
                             y_pred[..., None], delta=self.delta)
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


class BoxLoss(tf.keras.losses.Loss):
    def __init__(self, params, **kwargs):
        self.box_loss = SmoothL1Loss(params.loss.smooth_l1_loss.delta)
        super().__init__(**kwargs)

    def __call__(self, targets, predictions, sample_weight=None):
        loss = self.box_loss(targets, predictions)
        return loss


class ClassLoss(tf.keras.losses.Loss):
    def __init__(self, params, **kwargs):
        self.class_loss = FocalLoss(params.loss.focal_loss.alpha, params.loss.focal_loss.gamma)
        self._num_classes = params.architecture.num_classes
        super().__init__(**kwargs)

    def __call__(self, targets, predictions, sample_weight=None):
        loss = self.class_loss(
                tf.one_hot(tf.cast(targets, dtype=tf.int32),
                           depth=self._num_classes), predictions)
        return loss
