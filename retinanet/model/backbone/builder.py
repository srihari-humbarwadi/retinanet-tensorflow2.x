from retinanet.model.backbone.resnet import ResNet


def build_backbone(input_shape, params, normalization_op_params=None):
    if params.type == 'resnet':
        model = ResNet(
            input_shape=input_shape,
            depth=params.depth,
            checkpoint=params.checkpoint,
            normalization_op_params=normalization_op_params)

    else:
        raise ValueError('{} backbone not implemented'.format(
            params.type))
    return model
