import tensorflow as tf

from retinanet.losses.loss_impl import BoxLoss, ClassLoss


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

        return {
            'box-loss':
            box_loss,
            'class-loss':
            class_loss,
            'weighted-loss':
            self._box_loss_weight * box_loss +
            self._class_loss_weight * class_loss
        }
