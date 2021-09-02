import functools

from retinanet.model.layers.feature_fusion import FeatureFusion
from retinanet.model.layers.nearest_upsampling import NearestUpsampling2D
from retinanet.model.neck.fpn_base import FPNBase
from retinanet.model.utils import Identity


class FPN(FPNBase):

    def __init__(self,
                 filters=256,
                 min_level=3,
                 max_level=7,
                 backbone_max_level=5,
                 fusion_mode=None,
                 conv_2d_op_params=None,
                 normalization_op_params=None,
                 activation_fn=None,
                 **kwargs):

        if activation_fn is None:
            raise ValueError('`activation_fn` cannot be None')

        super(FPN, self).__init__(
            filters=filters,
            min_level=min_level,
            max_level=max_level,
            backbone_max_level=backbone_max_level,
            conv_2d_op_params=conv_2d_op_params,
            normalization_op_params=normalization_op_params,
            **kwargs)

        self.fusion_mode = fusion_mode
        self.upsample_op = functools.partial(NearestUpsampling2D, scale=2)

        self.channel_normalize_convs = {}
        self.channel_normalize_norms = {}
        self.output_convs = {}
        self.output_norms = {}
        self.output_activations = {}
        self.fusion_ops = {}
        self.fusion_activation_ops = {}

        for level in range(min_level, backbone_max_level + 1):
            level = str(level)
            self.channel_normalize_convs[level] = self._conv_2d_op(
                filters=self.filters,
                kernel_size=1,
                strides=1,
                padding='same',
                name='p{}-in-channel-normalize-conv-1x1'.format(level),
                **self._kernel_initializer_config)
            self.channel_normalize_norms[level] = self._normalization_op(
                name='p{}-in-channel-normalize-batch_normalization'.format(level))

        for level in range(min_level, max_level + 1):
            level = str(level)

            self.output_convs[level] = self._conv_2d_op(
                filters=self.filters,
                kernel_size=3,
                padding='same',
                strides=1,
                use_bias=conv_2d_op_params.use_bias_before_bn,
                name='p{}-out-conv-3x3'.format(level),
                **self._kernel_initializer_config)

            self.output_norms[level] = self._normalization_op(
                name='p{}-out-batch_normalization'.format(level))

            if int(level) != min_level:
                self.fusion_ops[level] = FeatureFusion(
                    mode=fusion_mode,
                    filters=filters,
                    name='p{}-in-fusion-with-p{}-in-upsampled'.format(
                        str(int(level) - 1), level))
                self.fusion_activation_ops[level] = activation_fn(
                    name='p{}-in-fusion-with-p{}-in-upsampled'.format(
                        str(int(level) - 1), level))

    def call(self, features, training=None):
        outputs = super(FPN, self).call(features, training=training)

        # normalize channel counts for backbone feature maps
        for level in range(self.min_level, self.backbone_max_level + 1):
            level = str(level)
            conv_layer = self.channel_normalize_convs[level]
            norm_layer = self.channel_normalize_norms[level]
            x = conv_layer(outputs[level])
            outputs[level] = norm_layer(x, training=training)

        # add top down pathway
        for level in range(self.max_level, self.min_level, -1):
            level = str(level)
            name = 'p{}-in-upsampled'.format(level)
            x = self.fusion_ops[level]([outputs[str(int(level) - 1)],
                                        self.upsample_op(name=name)(outputs[level])])
            outputs[str(int(level) - 1)] = self.fusion_activation_ops[level](x)

        # add output convs
        for level in range(self.min_level, self.max_level + 1):
            level = str(level)
            x = self.output_convs[level](outputs[level])
            x = self.output_norms[level](x, training=training)
            outputs[level] = Identity(name='p{}-out'.format(level))(x)

        return outputs
