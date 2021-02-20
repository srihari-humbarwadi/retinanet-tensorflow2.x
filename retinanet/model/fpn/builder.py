from retinanet.model.fpn.fpn import FPN


def build_fpn(params, conv_2d_op_params=None, normalization_op_params=None):
    if params.type == 'default':
        fpn = FPN(
            filters=params.filters,
            min_level=params.min_level,
            max_level=params.max_level,
            backbone_max_level=params.backbone_max_level,
            fusion_mode=params.fusion_mode,
            use_residual_connections=params.use_residual_connections,
            conv_2d_op_params=conv_2d_op_params,
            normalization_op_params=normalization_op_params,
            name='fpn_default')

    else:
        raise ValueError('{} FPN not implemented'.format(params.type))

    return fpn
