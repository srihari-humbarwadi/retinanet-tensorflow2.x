import json
import re

import tensorflow as tf
from absl import logging

from retinanet.losses import RetinaNetLoss
from retinanet.model.backbone import build_backbone
from retinanet.model.fpn import build_fpn
from retinanet.model.head import build_heads
from retinanet.model.layers import (FilterTopKDetections, FuseDetections,
                                    GenerateDetections,
                                    TransformBoxesAndScores)
from retinanet.optimizers import build_optimizer


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
            r'^(?!((fpn)|(box-head)|(class-head))).*(conv2d(_fixed_padding)?(|_([1-9]|10))|(sync_)?batch_normalization(|_([1-9]|10)))\/')  # noqa: E501
    }

    def __init__(self, params):
        self.params = params

    def __call__(self):
        input_shape = self.params.input.input_shape + [self.params.input.channels]
        images = tf.keras.Input(shape=input_shape, name="images")

        backbone = build_backbone(
            input_shape=input_shape,
            params=self.params.architecture.backbone,
            normalization_op_params=self.params.architecture.batch_norm)

        fpn = build_fpn(params=self.params.architecture.fpn,
                        conv_2d_op_params=self.params.architecture.conv_2d,
                        normalization_op_params=self.params.architecture.batch_norm)

        box_head, class_head = build_heads(
            params=self.params.architecture.head,
            min_level=self.params.architecture.fpn.min_level,
            max_level=self.params.architecture.fpn.max_level,
            conv_2d_op_params=self.params.architecture.conv_2d,
            normalization_op_params=self.params.architecture.batch_norm)

        features = backbone(images)
        features = fpn(features)

        for k, v in features.items():
            print(k, ':', v)

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

        inference_model = self.add_post_processing_stage(model)
        return inference_model

    def add_post_processing_stage(self, model, skip_nms=False):
        logging.info('Postprocessing stage config:\n{}'
                     .format(json.dumps(self.params.inference, indent=4)))

        x = FuseDetections(
            min_level=self.params.architecture.fpn.min_level,
            max_level=self.params.architecture.fpn.max_level)(model.output)

        x = TransformBoxesAndScores(params=self.params)(x)

        if not skip_nms:

            if self.params.inference.pre_nms_top_k > 0:

                x = FilterTopKDetections(
                    top_k=self.params.inference.pre_nms_top_k,
                    filter_per_class=self.params.inference.filter_per_class)(x)

            x = GenerateDetections(
                iou_threshold=self.params.inference.iou_threshold,
                score_threshold=self.params.inference.score_threshold,
                max_detections=self.params.inference.max_detections,
                soft_nms_sigma=self.params.inference.soft_nms_sigma,
                num_classes=self.params.architecture.head.num_classes,
                mode=self.params.inference.mode)(x)

        model = tf.keras.Model(inputs=[model.input], outputs=x)

        for layer in model.layers:
            logging.debug('Layer Name: {} | Output Shape: {}'
                          .format(layer.name, layer.output_shape))

        return model
