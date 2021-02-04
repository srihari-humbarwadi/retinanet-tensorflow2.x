import re

import tensorflow as tf
from absl import logging

from retinanet.model.backbone import build_backbone
from retinanet.model.fpn import build_fpn
from retinanet.model.layers import DecodePredictions
from retinanet.model.head import build_heads
from retinanet.optimizers import build_optimizer
from retinanet.losses import RetinaNetLoss


class ModelBuilder:

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
            r'^(conv2d(|_([1-9]|10))|(sync_)?batch_normalization(|_([1-9]|10)))\/')
    }

    def __init__(self, params):
        self.params = params

    def __call__(self):
        input_shape = self.params.input.input_shape + [self.params.input.channels]
        images = tf.keras.Input(shape=input_shape, name="images")

        backbone = build_backbone(
            input_shape=input_shape,
            params=self.params.architecture.backbone)

        fpn = build_fpn(params=self.params.architecture.fpn)

        box_head, class_head = build_heads(
            params=self.params.architecture.head,
            min_level=self.params.architecture.fpn.min_level,
            max_level=self.params.architecture.fpn.max_level)

        features = backbone(images)
        features = fpn(features)
        box_outputs = box_head(features)
        class_outputs = class_head(features)

        model = tf.keras.Model(
            inputs=[images],
            outputs={
                'class-predictions': class_outputs,
                'box-predictions': box_outputs
            },
            name='retinanet'
        )

        optimizer = build_optimizer(
            self.params.training.optimizer,
            precision=self.params.floatx.precision)

        _loss_fn = RetinaNetLoss(
            self.params.architecture.head.num_classes,
            self.params.loss)

        model.compile(optimizer=optimizer, loss=_loss_fn)

        return model

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
            key = str(i)
            class_predictions += [
                tf.keras.layers.Reshape(
                    [-1, self.params.architecture.head.num_classes])(
                    model.output['class-predictions'][key])
            ]
            box_predictions += [
                tf.keras.layers.Reshape([-1, 4])(
                    model.output['box-predictions'][key])
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
            key = str(i)
            class_predictions += [
                tf.keras.layers.Reshape([-1, self.params.architecture.num_classes])(
                    model.output['class-predictions'][key])
            ]
            box_predictions += [
                tf.keras.layers.Reshape([-1, 4])(
                    model.output['box-predictions'][key])
            ]
        class_predictions = tf.concat(class_predictions, axis=1)
        box_predictions = tf.concat(box_predictions, axis=1)
        predictions = (box_predictions, class_predictions)
        inference_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=predictions,
                                         name='model_with_fused_outputs')
        return inference_model
