import os
import json

import onnx
import onnx_graphsurgeon as gs
import onnxsim
import tf2onnx
from absl import logging
import numpy as np

from retinanet.dataloader.anchor_generator import AnchorBoxGenerator


def _add_nms_plugin(model, params):
    min_level = params.architecture.feature_fusion.min_level
    max_level = params.architecture.feature_fusion.max_level
    inference_params = params.inference

    anchor_boxes = AnchorBoxGenerator(
        *params.input.input_shape,
        min_level,
        max_level,
        params.anchor_params).boxes
    anchor_boxes = np.expand_dims(anchor_boxes.numpy(), axis=0)
    anchor_boxes = gs.Constant('anchor-boxes', anchor_boxes)

    gs_graph = gs.import_onnx(model)

    class_logits, raw_boxes = gs_graph.outputs
    class_logits.name = 'class-logits'
    raw_boxes.name = 'raw-boxes'
    batch_size = class_logits.shape[0]
    nms_inputs = [raw_boxes, class_logits, anchor_boxes]

    for tensor in nms_inputs:
        logging.info('NMS input name:{} | shape: {}'.format(
            tensor.name, tensor.shape))

    nms_plugin_attributes = {
        'plugin_version': '1',
        'background_class': -1,
        'max_output_boxes': inference_params['max_detections'],
        'score_threshold': inference_params['score_threshold'],
        'iou_threshold': inference_params['iou_threshold'],
        'score_activation': True,
        'box_coding': 1,
    }

    valid_detections = gs.Variable(
        name='valid_detections',
        dtype=np.int32,
        shape=[batch_size, 1])

    boxes = gs.Variable(
        name='detection_boxes',
        dtype=np.float32,
        shape=[batch_size, inference_params['max_detections'], 4])

    scores = gs.Variable(
        name='detection_scores',
        dtype=np.float32,
        shape=[batch_size, inference_params['max_detections']])

    classes = gs.Variable(
        name='detection_classes',
        dtype=np.int32,
        shape=[batch_size, inference_params['max_detections']])

    nms_outputs = [valid_detections, boxes, scores, classes]

    gs_graph.layer(
        op='EfficientNMS_TRT',
        name="non_maximum_suppression",
        inputs=nms_inputs,
        outputs=nms_outputs,
        attrs=nms_plugin_attributes)

    gs_graph.outputs = nms_outputs

    gs_graph.cleanup().toposort()
    model = gs.export_onnx(gs_graph)

    logging.info('Adding `EfficientNMS_TRT` pluging with the attributes:\n{}'.format(
        json.dumps(nms_plugin_attributes, indent=4)))

    return model


def save_concrete_function(
        function,
        input_signature,
        add_nms_plugin,
        opset,
        output_dir,
        target='tensorrt',
        model_params=None,
        simplify=True,
        large_model=False,
        debug=False):

    if add_nms_plugin and model_params is None:
        raise ValueError('model_params are required to add NMS plugin')

    tf2onnx.logging.basicConfig(
        level=tf2onnx.logging.get_verbosity_level(2 if debug else 1))

    onnx_model, _ = tf2onnx.convert.from_function(
        function=function,
        input_signature=input_signature,
        opset=opset,
        custom_ops=None,
        custom_op_handlers=None,
        custom_rewriter=None,
        inputs_as_nchw=None,
        extra_opset=None,
        shape_override=None,
        target=target,
        large_model=large_model,
        output_path=None)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'model.onnx')

    if simplify:
        logging.info('Running ONNX simplifier')
        onnx_model, status = onnxsim.simplify(onnx_model, check_n=3)
        if not status:
            raise AssertionError('Failed to simplify ONNX model')

    if add_nms_plugin:
        logging.info('Adding `EfficientNMS_TRT` plugin')
        onnx_model = _add_nms_plugin(onnx_model, model_params)

    onnx.save_model(onnx_model, output_path)
    logging.info('Saving ONNX model to: {}'.format(output_path))
