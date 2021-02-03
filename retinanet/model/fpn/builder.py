from retinanet.model.fpn.fpn import FPN


def build_fpn(params):
    if params.type == 'default':
        fpn = FPN(
            params.filters,
            params.min_level,
            params.max_level,
            params.backbone_max_level)

    else:
        raise ValueError('{} FPN not implemented'.format(params.type))

    return fpn
