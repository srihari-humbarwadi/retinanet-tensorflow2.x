import json
import re

import numpy as np
import tensorflow as tf
from absl import logging

from retinanet.core.layers.decode import DecodePredictions
from retinanet.core.utils import get_optimizer
from retinanet.utils.registry import Registry

# registry for internal modules.
NECK = Registry("neck")
BACKBONE = Registry("backbone")
HEAD = Registry("head")
LOSS = Registry("loss")
DETECTOR = Registry("detector")

# TODO: Skip Mixins, Implement separate class for Postprocessing module builder once
# nms ops are finalized


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

    FREEZE_VARS_REGEX = {
        'backbone': re.compile(r'^(?!((fpn)|(box-head)|(class-head)))'),
        'backbone-bn': re.compile(
            r'^(?!((fpn)|(box-head)|(class-head))).*(batch_normalization)'),
        'fpn': re.compile(r'^(fpn)'),
        'fpn-bn': re.compile(r'^(fpn).*(batch_normalization)'),
        'head': re.compile(r'^((box-head)|(class-head))(?!.*prediction)'),
        'head-bn': re.compile(r'^((box-head)|(class-head)).*(batch_normalization)'),
        'bn': re.compile(r'(batch_normalization)'),
        'resnet_initial': re.compile(
            r'^(conv2d(|_([1-9]|10))|batch_normalization(|_([1-9]|10)))\/')
    }

    def __init__(self, params):
        self.params = params
        self.detector_class = DETECTOR.get(params.architecture.detector)
        self.backbone_class = BACKBONE.get(params.architecture.backbone.type)
        self.neck_class = NECK.get(params.architecture.neck.type)

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

        backbone = self.backbone_class(
            self.input_shape,
            self.params.architecture.backbone.depth,
            checkpoint_dir=self.params.architecture.backbone.checkpoint)

        if self.params.fine_tuning.fine_tune and \
                self.params.fine_tuning.freeze_backbone:
            logging.warning('Freezing backbone for fine tuning')

            for layer in backbone.layers:
                layer.trainable = False

        neck = self.neck_class(
            self.params.architecture.neck.filters,
            self.params.architecture.neck.min_level,
            self.params.architecture.neck.max_level,
            self.params.architecture.neck.backbone_max_level)

        prior_prob_init = \
            tf.constant_initializer(-np.log((1 - 0.01) / 0.01))

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

        model = self.detector_class(backbone, neck, box_head, class_head)
        optimizer = get_optimizer(
            self.params.training.optimizer,
            precision=self.params.floatx.precision)
        logging.info(
            'Optimizer Config: \n{}'
            .format(json.dumps(optimizer.get_config(), indent=4)))

        if self.params.fine_tuning.fine_tune:
            logging.info(
                'Loading pretrained weights for fine-tuning from {}'.format(
                    self.params.fine_tuning.pretrained_checkpoint))
            model.load_weights(self.params.fine_tuning.pretrained_checkpoint,
                               skip_mismatch=True,
                               by_name=True)

        _loss_fn = self.loss_fn(
            self.params.architecture.num_classes, self.params.loss)
        model.compile(optimizer=optimizer, loss=_loss_fn)
        return model
