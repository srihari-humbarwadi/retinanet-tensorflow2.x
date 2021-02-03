from retinanet.model.backbone.resnet import ResNet


def build_backbone(input_shape, params):
    if params.type == 'resnet':
        model = ResNet(
            input_shape=input_shape,
            depth=params.depth,
            checkpoint=params.checkpoint)

    else:
        raise ValueError('{} backbone not implemented'.format(
            params.type))
    return model
