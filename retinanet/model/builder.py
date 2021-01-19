import json
import re

import tensorflow as tf
from absl import logging
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from retinanet.losses import RetinaNetLoss
from retinanet.model.decode_predictions import DecodePredictions
from retinanet.model.retinanet import retinanet_builder
from retinanet.model.utils import add_l2_regularization, get_optimizer


def model_builder(params):
    def _model_fn():
        input_shape = params.input.input_shape + [params.input.channels]
        model = retinanet_builder(input_shape, params.architecture)

        logging.info('Trainable weights: {}'.format(
            len(model.trainable_weights)))

        if params.architecture.freeze_initial_layers:
            freeze_pattern = \
                r'(conv2d(|_([1-9]|10))|batch_normalization(|_([1-9]|10)))\/'
            logging.info('Freezing initial weights')
            for layer in model.layers:
                for weight in layer.weights:
                    if re.search(freeze_pattern, weight.name) \
                            is not None and layer.trainable:
                        layer.trainable = False
            logging.info('Trainable weights after freezing: {}'.format(
                len(model.trainable_weights)))

        if params.architecture.use_weight_decay:
            alpha = params.architecture.weight_decay_alpha

            for layer in model.layers:
                if isinstance(layer,
                              tf.keras.layers.Conv2D) and layer.trainable:
                    model.add_loss(add_l2_regularization(layer.kernel, alpha))

            logging.info('Initial l2_regularization loss {}'.format(
                tf.math.add_n(model.losses).numpy()))

        optimizer = get_optimizer(params.training.optimizer)
        logging.info(
            'Optimizer Config: {}'
            .format(json.dumps(
                tf.keras.utils.serialize_keras_object(optimizer), indent=4)))

        if params.floatx.precision == 'mixed_float16':
            logging.info(
                'Wrapping optimizer with `LossScaleOptimizer` for AMP training'
            )
            optimizer = mixed_precision.LossScaleOptimizer(
                optimizer, loss_scale='dynamic')

        if params.fine_tuning.fine_tune:
            logging.info(
                'Loading pretrained weights for fine-tuning from {}'.format(
                    params.fine_tuning.pretrained_checkpoint))
            model.load_weights(params.fine_tuning.pretrained_checkpoint,
                               skip_mismatch=True,
                               by_name=True)

            logging.info(
                'l2_regularization loss after loading pretrained \
                    weights{}'.format(tf.math.add_n(model.losses).numpy()))

        loss_fn = RetinaNetLoss(params.architecture.num_classes, params.loss)
        model.compile(optimizer=optimizer, loss=loss_fn)

        model.summary(print_fn=logging.debug)

        logging.info('Total trainable parameters: {:,}'.format(
            sum([
                tf.keras.backend.count_params(x)
                for x in model.trainable_variables
            ])))
        return model

    return _model_fn


def _add_post_processing_stage(model, params):
    class_predictions = []
    box_predictions = []
    for i in range(3, 8):
        class_predictions += [
            tf.keras.layers.Reshape([
                -1, params.architecture.num_classes
            ])(model.output['class-predictions'][i])
        ]
        box_predictions += [
            tf.keras.layers.Reshape(
                [-1, 4])(model.output['box-predictions'][i])
        ]
    class_predictions = tf.concat(class_predictions, axis=1)
    box_predictions = tf.concat(box_predictions, axis=1)
    predictions = (box_predictions, class_predictions)
    detections = DecodePredictions(params)(predictions)
    inference_model = tf.keras.Model(inputs=model.inputs,
                                     outputs=detections,
                                     name='retinanet_inference')
    return inference_model


def make_eval_model(model, params):
    eval_model = _add_post_processing_stage(model, params)

    logging.info('Created inference model with params: {}'
                 .format(params.inference))
    return eval_model


def prepare_model_for_export(trainer):
    model = trainer.model
    params = trainer.params

    model.optimizer = None
    model.compiled_loss = None
    model.compiled_metrics = None
    model._metrics = []

    inference_model = _add_post_processing_stage(model, params)

    logging.info('Created inference model with params: {}'
                 .format(params.inference))
    return inference_model


def _fuse_model_outputs(model, params):
    class_predictions = []
    box_predictions = []
    for i in range(3, 8):
        class_predictions += [
            tf.keras.layers.Reshape([
                -1, params.architecture.num_classes
            ])(model.output['class-predictions'][i])
        ]
        box_predictions += [
            tf.keras.layers.Reshape(
                [-1, 4])(model.output['box-predictions'][i])
        ]
    class_predictions = tf.concat(class_predictions, axis=1)
    box_predictions = tf.concat(box_predictions, axis=1)
    predictions = (box_predictions, class_predictions)
    inference_model = tf.keras.Model(inputs=model.inputs,
                                     outputs=predictions,
                                     name='model_with_fused_outputs')
    return inference_model
