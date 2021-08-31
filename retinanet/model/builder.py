import json
import re

import tensorflow as tf
from absl import logging
from retinanet.losses import RetinaNetLoss
from retinanet.model.backbone import build_backbone
from retinanet.model.neck import build_neck
from retinanet.model.head import build_detection_heads, build_auxillary_head
from retinanet.model.layers import (BalanceFeatures, FilterTopKDetections,
                                    FuseDetections, GenerateDetections,
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
            r'^(?!((fpn)|((stacked_)?mlaf)|(box-head)|(class-head))).*(conv2d(_fixed_padding)?(|_([1-9]|10))|(sync_)?batch_normalization(|_([1-9]|10)))\/')  # noqa: E501
    }

    def __init__(self, params):
        self.params = params

    def __call__(self):
        params = self.params
        input_shape = params.input.input_shape + [params.input.channels]
        images = tf.keras.Input(shape=input_shape, name="images")

        backbone = build_backbone(
            input_shape=input_shape,
            params=params.architecture.backbone,
            normalization_op_params=params.architecture.batch_norm)

        neck = build_neck(
            params=params.architecture.feature_fusion,
            conv_2d_op_params=params.architecture.conv_2d,
            normalization_op_params=params.architecture.batch_norm)

        box_head, class_head = build_detection_heads(
            params=params.architecture.head,
            min_level=params.architecture.feature_fusion.min_level,
            max_level=params.architecture.feature_fusion.max_level,
            conv_2d_op_params=params.architecture.conv_2d,
            normalization_op_params=params.architecture.batch_norm)

        auxillary_head = None
        if params.architecture.auxillary_head.use_auxillary_head:
            auxillary_head = build_auxillary_head(
                num_convs=params.architecture.auxillary_head.num_convs,
                filters=params.architecture.auxillary_head.filters,
                num_anchors=params.architecture.head.num_anchors,
                min_level=params.architecture.feature_fusion.min_level,
                max_level=params.architecture.feature_fusion.max_level,
                conv_2d_op_params=params.architecture.conv_2d,
                normalization_op_params=params.architecture.batch_norm)

        features = backbone(images)
        features = neck(features)

        if params.architecture.feature_fusion.use_balanced_features:
            features = BalanceFeatures(
                min_level=params.architecture.feature_fusion.min_level,
                max_level=params.architecture.feature_fusion.max_level,
                intermediate_level=params.architecture.feature_fusion.min_level+1)(features)  # noqa: E501

        box_outputs = box_head(features)
        class_outputs = class_head(features)

        outputs = {
            'class-predictions': class_outputs,
            'box-predictions': box_outputs
            }

        if auxillary_head is not None:
            auxillary_outputs = auxillary_head(features)
            outputs['iou-predictions'] = auxillary_outputs

        model = tf.keras.Model(
            inputs=[images],
            outputs=outputs,
            name='retinanet')

        optimizer = build_optimizer(
            params.training.optimizer,
            params.training.train_steps,
            precision=params.floatx.precision)

        _loss_fn = RetinaNetLoss(
            params.architecture.head.num_classes,
            params.loss)

        model.compile(optimizer=optimizer, loss=_loss_fn)

        return model

    def prepare_model_for_export(self, model, skip_nms=False):
        model.optimizer = None
        model.compiled_loss = None
        model.compiled_metrics = None
        model._metrics = []

        inference_model = self.add_post_processing_stage(
            model=model, skip_nms=skip_nms)
        return inference_model

    def add_post_processing_stage(self, model, skip_nms=False):
        params = self.params
        logging.info('Postprocessing stage config:\n{}'
                     .format(json.dumps(params.inference, indent=4)))

        x = FuseDetections(
            min_level=params.architecture.feature_fusion.min_level,
            max_level=params.architecture.feature_fusion.max_level)(model.output)

        x = TransformBoxesAndScores(params=params)(x)

        if params.inference.pre_nms_top_k > 0 and not skip_nms:
            x = FilterTopKDetections(
                top_k=params.inference.pre_nms_top_k,
                filter_per_class=params.inference.filter_per_class)(x)
        else:
            logging.warning('Skipping top-k filtering !!!')

        if not skip_nms:
            x = GenerateDetections(
                iou_threshold=params.inference.iou_threshold,
                score_threshold=params.inference.score_threshold,
                max_detections=params.inference.max_detections,
                soft_nms_sigma=params.inference.soft_nms_sigma,
                num_classes=params.architecture.head.num_classes,
                mode=params.inference.mode)(x)
        else:
            logging.warning('Skipping NMS filtering !!!')

        model = tf.keras.Model(inputs=[model.input], outputs=x)
        for layer in model.layers:
            logging.debug('Layer Name: {} | Output Shape: {}'
                          .format(layer.name, layer.output_shape))

        return model
