import tensorflow as tf
from absl import logging

from retinanet.model.backbone.resnet import resnet_builder


def backbone_builder(input_shape, params):
    if params.architecture.backbone.type == 'resnet':
        model = resnet_builder(
            input_shape,
            params.architecture.backbone.depth,
            params.architecture.backbone.checkpoint)

    if params.fine_tuning.fine_tune:
        if params.fine_tuning.freeze_backbone:
            logging.warning('Freezing backbone for fine tuning')

            for layer in model.layers:
                if isinstance(layer, (
                        tf.keras.layers.Conv2D,
                        tf.keras.layers.BatchNormalization,
                        tf.keras.layers.experimental.SyncBatchNormalization)):
                    layer.trainable = False

    return model
