import json
import re

import numpy as np
import tensorflow as tf
from absl import logging
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from retinanet.core.layers.decode import DecodePredictions
from retinanet.core.utils import add_l2_regularization, get_optimizer
from retinanet.utils.registry import Registry

# registry for internal modules.
NECK = Registry("neck")
BACKBONE = Registry("backbone")
HEAD = Registry("head")
LOSS = Registry("loss")
DETECTOR = Registry("detector")

class BuilderMixin:

    def prepare_model_for_export(self, model):
        model.optimizer = None
        model.compiled_loss = None
        model.compiled_metrics = None
        model._metrics = []

        inference_model = self._add_post_processing_stage(model)
        _ = model(tf.random.uniform(shape=[1, 640, 640, 3]))
        return inference_model

    def _add_post_processing_stage(self, model):
        class_predictions = []
        box_predictions = []
        for i in range(3, 8):
            predictions_key = 'p{}-predictions'.format(i)

            class_predictions += [
                tf.keras.layers.Reshape([-1, self.params.architecture.num_classes])(
                    model.output['class-predictions'][predictions_key])
            ]
            box_predictions += [
                tf.keras.layers.Reshape([-1, 4])(
                    model.output['box-predictions'][predictions_key])
            ]

        class_predictions = tf.concat(class_predictions, axis=1)
        box_predictions = tf.concat(box_predictions, axis=1)
        predictions = (box_predictions, class_predictions)
        detections = DecodePredictions(self.params)(predictions)
        inference_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=detections,
                                         name='retinanet_inference')
        logging.info('Created inference model with params: {}'
                     .format(self.params.inference))
        return inference_model

    def make_eval_model(self, model):
        eval_model = self._add_post_processing_stage(model)
        return eval_model

    def _fuse_model_outputs(self, model):
        class_predictions = []
        box_predictions = []
        for i in range(3, 8):
            predictions_key = 'p{}-predictions'.format(i)

            class_predictions += [
                tf.keras.layers.Reshape([-1, self.params.architecture.num_classes])(
                    model.output['class-predictions'][predictions_key])
            ]
            box_predictions += [
                tf.keras.layers.Reshape([-1, 4])(
                    model.output['box-predictions'][predictions_key])
            ]
        class_predictions = tf.concat(class_predictions, axis=1)
        box_predictions = tf.concat(box_predictions, axis=1)
        predictions = (box_predictions, class_predictions)
        inference_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=predictions,
                                         name='model_with_fused_outputs')
        return inference_model

class ModelBuilder(BuilderMixin):
    """ builds detector model using config. """

    def __init__(self, params):
        self.params = params
        self.detector_class = DETECTOR.get(params.architecture.detector)
        self.backbone_class = BACKBONE.get(params.architecture.backbone.type)
        self.fpn_class = NECK.get(params.architecture.neck.type)

        # TODO (kartik4949): remove hardcoded head
        self.head_class = HEAD.get('retinanet_detection_head')

        # TODO (kartik4949): remove hardcoded loss
        self.loss_fn = LOSS.get("retinanet")
        self.input_shape = self.params.input.input_shape + \
            [self.params.input.channels]
        self.input_layer = tf.keras.Input(shape=self.input_shape, name="image_input")

    def __call__(self):

        num_classes = self.params.architecture.num_classes

        num_scales = len(self.params.anchor_params.scales)
        num_aspect_rations = len(self.params.anchor_params.aspect_ratios)

        num_anchors = num_scales * num_aspect_rations

        prior_prob_init = \
            tf.constant_initializer(-np.log((1 - 0.01) / 0.01))

        backbone = self.backbone_class(
            self.input_shape,
            self.params.architecture.backbone.depth,
            checkpoint_dir=self.params.architecture.backbone.checkpoint)

        if self.params.fine_tuning.fine_tune and \
                self.params.fine_tuning.freeze_backbone:
            logging.warning('Freezing backbone for fine tuning')

            for layer in backbone.layers:
                layer.trainable = False

        fpn = self.fpn_class(
            self.params.architecture.neck.filters,
            self.params.architecture.neck.min_level,
            self.params.architecture.neck.max_level,
            self.params.architecture.neck.backbone_max_level)

        box_head = self.head_class(
            self.params.architecture.num_head_convs,
            self.params.architecture.num_head_filters,
            num_anchors * 4,
            name='box-head')

        class_head = self.head_class(
            self.params.architecture.num_head_convs,
            self.params.architecture.num_head_filters,
            num_anchors * num_classes,
            prediction_bias_initializer=prior_prob_init,
            name='class-head')

        model = self.detector_class(backbone, fpn, box_head, class_head)

        logging.info('Trainable weights: {}'.format(
            len(model.trainable_weights)))

        # TODO unify all weight freezing functionality
        if self.params.architecture.freeze_initial_layers:
            freeze_pattern = \
                r'(conv2d(|_([1-9]|10))|batch_normalization(|_([1-9]|10)))\/'
            logging.info('Freezing initial weights')

            for layer in model.layers[1].layers:
                for weight in layer.weights:
                    if re.search(freeze_pattern, weight.name) \
                            is not None and layer.trainable:
                        layer.trainable = False

            logging.info('Trainable weights after freezing: {}'.format(
                len(model.trainable_weights)))

        #  TODO avoid `model.add_loss`; Maintain a list of all variable names
        #  that need to be included in weight decay loss. Call tf.nn.l2_loss with
        #  variable namesinside per_replica_train step
        if self.params.training.use_weight_decay:
            alpha = self.params.architecture.weight_decay_alpha

            for layer in model.layers:
                if isinstance(layer,
                              tf.keras.layers.Conv2D) and layer.trainable:
                    model.add_loss(add_l2_regularization(layer.kernel, alpha))

                elif isinstance(layer, tf.keras.Model):
                    for inner_layer in layer.layers:
                        if isinstance(inner_layer, tf.keras.layers.Conv2D) and \
                                inner_layer.trainable:
                            model.add_loss(
                                add_l2_regularization(inner_layer.kernel, alpha))

            logging.info('Initial l2_regularization loss {}'.format(
                tf.math.add_n(model.losses).numpy()))

        optimizer = get_optimizer(self.params.training.optimizer)
        logging.info(
            'Optimizer Config: \n{}'
            .format(json.dumps(optimizer.get_config(), indent=4)))

        # TODO `get_optimizer` should handle this
        if self.params.floatx.precision == 'mixed_float16':
            logging.info(
                'Wrapping optimizer with `LossScaleOptimizer` for AMP training'
            )
            optimizer = mixed_precision.LossScaleOptimizer(
                optimizer, loss_scale='dynamic')

        if self.params.fine_tuning.fine_tune:
            logging.info(
                'Loading pretrained weights for fine-tuning from {}'.format(
                    self.params.fine_tuning.pretrained_checkpoint))
            model.load_weights(self.params.fine_tuning.pretrained_checkpoint,
                               skip_mismatch=True,
                               by_name=True)

            # TODO freeze batchnorm with unified interface for freezing weights
            if self.params.fine_tuning.freeze_batch_normalization:
                logging.warning('Freezing BatchNormalization layers for fine tuning')

                for layer in model.layers:
                    if isinstance(layer, (
                            tf.keras.layers.BatchNormalization,
                            tf.keras.layers.experimental.SyncBatchNormalization)):
                        layer.trainable = False

        if self.params.training.use_weight_decay:
            logging.debug(
                'l2_regularization loss after loading pretrained weights {}'
                .format(tf.math.add_n(model.losses).numpy()))

        _loss_fn = self.loss_fn(
            self.params.architecture.num_classes, self.params.loss)
        model.compile(optimizer=optimizer, loss=_loss_fn)

        model.summary(print_fn=logging.debug)

        logging.info('Total trainable parameters: {:,}'.format(
            sum([
                tf.keras.backend.count_params(x)
                for x in model.trainable_variables
            ])))
        return model
