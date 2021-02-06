from retinanet.model.fpn.fpn import FPN


def build_fpn(params, normalization_op_params=None):
    if params.type == 'default':
        fpn = FPN(
            filters=params.filters,
            min_level=params.min_level,
            max_level=params.max_level,
            backbone_max_level=params.backbone_max_level,
            normalization_op_params=normalization_op_params)

    else:
        raise ValueError('{} FPN not implemented'.format(params.type))

    return fpn
