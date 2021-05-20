from retinanet.model.neck.fpn import FPN
from retinanet.model.neck.multi_level_attention_fusion import \
    MultiLevelAttentionFusion


def build_neck(params, conv_2d_op_params=None, normalization_op_params=None):
    if params.type == 'fpn':
        neck = FPN(
            filters=params.filters,
            min_level=params.min_level,
            max_level=params.max_level,
            backbone_max_level=params.backbone_max_level,
            fusion_mode=params.fusion_mode,
            use_residual_connections=params.use_residual_connections,
            conv_2d_op_params=conv_2d_op_params,
            normalization_op_params=normalization_op_params,
            name='fpn')

    elif params.type == 'multi_level_attention':
        neck = MultiLevelAttentionFusion(
            filters=params.filters,
            projection_dim=params.projection_dim,
            min_level=params.min_level,
            max_level=params.max_level,
            backbone_max_level=params.backbone_max_level,
            conv_2d_op_params=conv_2d_op_params,
            normalization_op_params=normalization_op_params,
            name='mlaf')
    else:
        raise ValueError('{} FPN not implemented'.format(params.type))

    return neck
