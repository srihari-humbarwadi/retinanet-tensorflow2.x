from retinanet.model.neck.fpn import FPN
from retinanet.model.neck.multi_level_attention_fusion import \
    MultiLevelAttentionFusion
from retinanet.model.neck.stacked_multi_level_attention_fusion import \
    StackedMultiLevelAttentionFusion


def build_neck(
        params,
        conv_2d_op_params=None,
        normalization_op_params=None,
        activation_fn=None):

    if activation_fn is None:
        raise ValueError('`activation_fn` cannot be None')

    if params.type == 'fpn':
        neck = FPN(
            filters=params.filters,
            min_level=params.min_level,
            max_level=params.max_level,
            backbone_max_level=params.backbone_max_level,
            fusion_mode=params.fusion_mode,
            conv_2d_op_params=conv_2d_op_params,
            normalization_op_params=normalization_op_params,
            activation_fn=activation_fn,
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
            use_channel_attention=params.use_channel_attention,
            name='mlaf')

    elif params.type == 'stacked_multi_level_attention':
        neck = StackedMultiLevelAttentionFusion(
            filters=params.filters,
            projection_dim=params.projection_dim,
            num_repeats=params.num_repeats,
            min_level=params.min_level,
            max_level=params.max_level,
            backbone_max_level=params.backbone_max_level,
            conv_2d_op_params=conv_2d_op_params,
            normalization_op_params=normalization_op_params,
            use_channel_attention=params.use_channel_attention,
            name='stacked_mlaf')
    else:
        raise ValueError('{} FPN not implemented'.format(params.type))

    return neck
