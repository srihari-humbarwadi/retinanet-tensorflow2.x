import functools

import tensorflow as tf

from retinanet.model.neck.multi_level_attention_fusion import \
    MultiLevelAttentionFusion


class StackedMultiLevelAttentionFusion(tf.keras.layers.Layer):

    def __init__(self,
                 filters=256,
                 projection_dim=64,
                 num_repeats=2,
                 min_level=3,
                 max_level=7,
                 backbone_max_level=5,
                 conv_2d_op_params=None,
                 normalization_op_params=None,
                 use_channel_attention=True,
                 **kwargs):
        super(StackedMultiLevelAttentionFusion, self).__init__(**kwargs)

        self.num_repeats = num_repeats
        mlaf_block = functools.partial(
            MultiLevelAttentionFusion,
            filters=filters,
            projection_dim=projection_dim,
            min_level=min_level,
            backbone_max_level=backbone_max_level,
            conv_2d_op_params=conv_2d_op_params,
            normalization_op_params=normalization_op_params,
            use_channel_attention=use_channel_attention,
            **kwargs)

        self._blocks = {}

        for i in range(num_repeats):
            block_max_level = \
                max_level if (i == num_repeats - 1) else backbone_max_level

            self._blocks[str(i)] = \
                mlaf_block(
                    max_level=block_max_level,
                    use_lateral_conv=(i == 0),
                    name='mlaf_' + str(i+1))

    def call(self, features, training=False):
        outputs = features

        for i in range(self.num_repeats):
            outputs = self._blocks[str(i)](outputs, training=training)

        return outputs
