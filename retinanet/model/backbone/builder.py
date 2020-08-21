from retinanet.model.backbone.resnet import resnet_builder


def backbone_builder(input_shape, params):
    if params.type == 'resnet':
        return resnet_builder(input_shape, params.depth, params.checkpoint)
