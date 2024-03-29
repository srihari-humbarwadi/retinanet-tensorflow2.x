import tensorflow as tf
from absl import logging

from retinanet.losses.loss_impl import BoxLoss, ClassLoss, IouPredictionLoss


class RetinaNetLoss(tf.Module):
    def __init__(self, num_classes, params):
        self.box_loss = BoxLoss(params.smooth_l1_loss)
        self.class_loss = ClassLoss(num_classes, params.focal_loss)
        self.iou_prediction_loss = IouPredictionLoss()

        self._box_loss_weight = tf.convert_to_tensor(params.box_loss_weight)
        self._class_loss_weight = tf.convert_to_tensor(
            params.class_loss_weight)
        self._auxillary_loss_weight = tf.convert_to_tensor(
            params.auxillary_loss_weight)

        self._use_moving_average_normalizer = False

        if params.normalizer.use_moving_average:
            self._use_moving_average_normalizer = True
            self.normalizer_momentum = params.normalizer.momentum

            self.moving_average_normalizer = tf.Variable(
                0.0,
                name='moving_average_normalizer',
                dtype=tf.float32,
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN)

            logging.warning(
                'Using moving average loss normalizer with momentum: {}'
                .format(self.normalizer_momentum))

    def __call__(self, targets, predictions):
        normalizer = tf.reduce_sum(targets['num-positives']) + 1.0

        if self._use_moving_average_normalizer:
            normalizer = tf.keras.backend.moving_average_update(
                self.moving_average_normalizer,
                normalizer,
                self.normalizer_momentum)
        else:
            replica_context = tf.distribute.get_replica_context()
            normalizer = replica_context.all_reduce(
                tf.distribute.ReduceOp.SUM,
                normalizer) / replica_context.num_replicas_in_sync

        class_loss = self.class_loss(
            targets=targets['class-targets'],
            predictions=predictions['class-predictions'])

        box_loss = self.box_loss(
            targets=targets['box-targets'],
            predictions=predictions['box-predictions'])

        class_loss /= normalizer
        box_loss /= normalizer

        weighted_loss = self._box_loss_weight * box_loss + \
            self._class_loss_weight * class_loss

        losses = {
            'box-loss': box_loss,
            'class-loss': class_loss,
            'weighted-loss': weighted_loss,
            'num-anchors-matched': normalizer
        }

        if 'iou-predictions' in predictions:
            iou_prediction_loss = self.iou_prediction_loss(
                targets=targets['iou-targets'],
                predictions=predictions['iou-predictions'])
            iou_prediction_loss /= normalizer

            losses['weighted-loss'] += (self._auxillary_loss_weight *
                                        iou_prediction_loss)
            losses['iou-prediction-loss'] = iou_prediction_loss
        else:
            losses['iou-prediction-loss'] = 0.0
        return losses
