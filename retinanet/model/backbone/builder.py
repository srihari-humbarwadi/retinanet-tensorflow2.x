from copy import deepcopy
from retinanet.model.backbone.resnet import ResNet
from retinanet.model.backbone.efficientnet import EfficientNet


def build_backbone(input_shape, params, normalization_op_params=None):
    if 'resnet' in params.type.lower():
        resnet_params = deepcopy(params)
        resnet_params.pop('type')

        return ResNet(
            input_shape=input_shape,
            normalization_op_params=normalization_op_params,
            **resnet_params)

    if 'efficientnet' in params.type.lower():
        return EfficientNet(
            input_shape=input_shape,
            model_name=params.type,
            checkpoint=params.checkpoint,
            normalization_op_params=normalization_op_params,
            override_params=params.get('override_params', None))

    raise ValueError('{} backbone not implemented'.format(
        params.type))
