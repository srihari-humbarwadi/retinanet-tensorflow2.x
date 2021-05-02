from copy import deepcopy
from retinanet.model.backbone.resnet import ResNet


def build_backbone(input_shape, params, normalization_op_params=None):
    if params.type == 'resnet':
        resnet_params = deepcopy(params)
        resnet_params.pop('type')

        model = ResNet(
            input_shape=input_shape,
            normalization_op_params=normalization_op_params,
            **resnet_params)

    else:
        raise ValueError('{} backbone not implemented'.format(
            params.type))
    return model
