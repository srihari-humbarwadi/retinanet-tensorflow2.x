# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import functools

import numpy as np
import tensorflow as tf
from absl import logging

from retinanet.model.utils import get_normalization_op


def pad_to_multiple(tensor, multiple):
    batch, height, width, channels = tensor.get_shape().as_list()
    padded_height = tf.cast(tf.math.ceil(height / multiple) * multiple,
                            dtype=tf.int32)
    padded_width = tf.cast(tf.math.ceil(width / multiple) * multiple,
                           dtype=tf.int32)
    return tf.image.pad_to_bounding_box(tensor, 0, 0, padded_height,
                                        padded_width)


def _scale_filters(filters, multiplier, base=8):
    """Scale the filters accordingly to (multiplier, base)."""
    round_half_up = int(int(filters) * multiplier / base + 0.5)
    result = int(round_half_up * base)
    return max(result, base)


def _swish6(h):
    with tf.name_scope('swish6'):
        return h * tf.nn.relu6(h + np.float32(3)) * np.float32(1. / 6.)


def _conv(
    x,
    filters,
    kernel_size,
    strides=1,
    normalizer_fn=tf.keras.layers.BatchNormalization,
    activation_fn=tf.nn.relu6,
    normalization_op_params=None,
):
    if activation_fn is None:
        raise ValueError('Activation function cannot be None. Use tf.identity '
                         'instead to better support quantized training.')

    if normalization_op_params is None and normalizer_fn is not None:
        raise ValueError('Normalization params cannot be `None`')

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer=tf.initializers.VarianceScaling())(x)

    if normalizer_fn is not None:
        normaliztion_op = get_normalization_op(**normalization_op_params)
        x = normaliztion_op()(x)

    x = activation_fn(x)
    return x


def _separable_conv(x,
                    filters,
                    kernel_size,
                    strides=1,
                    activation_fn=tf.nn.relu6,
                    normalization_op_params=None):

    if activation_fn is None:
        raise ValueError('Activation function cannot be None. Use tf.identity '
                         'instead to better support quantized training.')
    # Depthwise variant of He initialization derived under the principle proposed
    # in the original paper. Note the original He normalization was designed for
    # full convolutions and calling tf.initializers.he_normal() can over-estimate
    # the fan-in of a depthwise kernel by orders of magnitude.
    stddev = (2.0 / kernel_size**2)**0.5 / .87962566103423978

    if normalization_op_params is None:
        raise ValueError('Normalization params cannot be `None`')

    if filters is not None:
        x = tf.keras.layers.SeparableConv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            depthwise_initializer=tf.initializers.TruncatedNormal(
                stddev=stddev),
            pointwise_initializer=tf.initializers.VarianceScaling())(x)

    else:
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            depth_multiplier=1,
            depthwise_initializer=tf.initializers.TruncatedNormal(
                stddev=stddev))(x)

    normaliztion_op = get_normalization_op(**normalization_op_params)
    x = normaliztion_op()(x)

    x = activation_fn(x)
    return x


def _squeeze_and_excite(x,
                        hidden_dim,
                        activation_fn=tf.nn.relu6,
                        normalization_op_params=None):

    if normalization_op_params is None:
        raise ValueError('Normalization params cannot be `None`')

    _, height, width, channels = x.get_shape().as_list()

    u = tf.keras.layers.AveragePooling2D([height, width],
                                         strides=1,
                                         padding='valid')(x)
    u = _conv(u,
              hidden_dim,
              1,
              normalizer_fn=None,
              activation_fn=activation_fn,
              normalization_op_params=normalization_op_params)
    u = _conv(u,
              channels,
              1,
              normalizer_fn=None,
              activation_fn=tf.nn.sigmoid,
              normalization_op_params=normalization_op_params)
    return u * x


def _inverted_bottleneck_no_expansion(x,
                                      filters,
                                      activation_fn=tf.nn.relu6,
                                      kernel_size=3,
                                      strides=1,
                                      use_se=False,
                                      normalization_op_params=None):
    """Inverted bottleneck layer without the first 1x1 expansion convolution."""

    if normalization_op_params is None:
        raise ValueError('Normalization params cannot be `None`')

    _, height, width, channels = x.get_shape().as_list()

    # Setting filters to None will make _separable_conv a depthwise conv.
    x = _separable_conv(x,
                        None,
                        kernel_size,
                        strides=strides,
                        activation_fn=activation_fn,
                        normalization_op_params=normalization_op_params)
    if use_se:
        hidden_dim = _scale_filters(channels, 0.25)
        x = _squeeze_and_excite(x,
                                hidden_dim,
                                activation_fn=activation_fn,
                                normalization_op_params=normalization_op_params)
    x = _conv(x,
              filters,
              1,
              activation_fn=tf.identity,
              normalization_op_params=normalization_op_params)
    return x


def _inverted_bottleneck(x,
                         filters,
                         activation_fn=tf.nn.relu6,
                         kernel_size=3,
                         expansion=8,
                         strides=1,
                         use_se=False,
                         residual=True,
                         normalization_op_params=None):
    """Inverted bottleneck layer."""

    if normalization_op_params is None:
        raise ValueError('Normalization params cannot be `None`')

    _, height, width, channels = x.get_shape().as_list()
    shortcut = x
    expanded_filters = channels * expansion

    if expansion <= 1:
        raise ValueError('Expansion factor must be greater than 1.')

    x = _conv(x,
              expanded_filters,
              1,
              activation_fn=activation_fn,
              normalization_op_params=normalization_op_params)

    # Setting filters to None will make _separable_conv a depthwise conv.
    x = _separable_conv(x,
                        None,
                        kernel_size,
                        strides=strides,
                        activation_fn=activation_fn,
                        normalization_op_params=normalization_op_params)
    if use_se:
        hidden_dim = _scale_filters(expanded_filters, 0.25)
        x = _squeeze_and_excite(x,
                                hidden_dim,
                                activation_fn=activation_fn,
                                normalization_op_params=normalization_op_params)
    x = _conv(x,
              filters,
              1,
              activation_fn=tf.identity,
              normalization_op_params=normalization_op_params)
    if residual:
        x = x + shortcut
    return x


def _fused_conv(x,
                filters,
                activation_fn=tf.nn.relu6,
                kernel_size=3,
                expansion=8,
                strides=1,
                use_se=False,
                residual=True,
                normalization_op_params=None):
    """Fused convolution layer."""

    if normalization_op_params is None:
        raise ValueError('Normalization params cannot be `None`')

    _, height, width, channels = x.get_shape().as_list()
    shortcut = x
    expanded_filters = channels * expansion

    if expansion <= 1:
        raise ValueError('Expansion factor must be greater than 1.')

    x = _conv(x,
              expanded_filters,
              kernel_size,
              strides=strides,
              activation_fn=activation_fn,
              normalization_op_params=normalization_op_params)
    if use_se:
        hidden_dim = _scale_filters(expanded_filters, 0.25)
        x = _squeeze_and_excite(x,
                                hidden_dim,
                                activation_fn=activation_fn,
                                normalization_op_params=normalization_op_params)
    x = _conv(x,
              filters,
              1,
              activation_fn=tf.identity,
              normalization_op_params=normalization_op_params)
    if residual:
        x = x + shortcut
    return x


def _tucker_conv(x,
                 filters,
                 activation_fn=tf.nn.relu6,
                 kernel_size=3,
                 input_rank_ratio=0.25,
                 output_rank_ratio=0.25,
                 strides=1,
                 residual=True,
                 normalization_op_params=None):
    """Tucker convolution layer (generalized bottleneck)."""

    if normalization_op_params is None:
        raise ValueError('Normalization params cannot be `None`')

    _, height, width, channels = x.get_shape().as_list()
    shortcut = x
    input_rank = _scale_filters(channels, input_rank_ratio)

    x = _conv(x,
              input_rank,
              1,
              activation_fn=activation_fn,
              normalization_op_params=normalization_op_params)
    output_rank = _scale_filters(filters, output_rank_ratio)
    x = _conv(x,
              output_rank,
              kernel_size,
              strides=strides,
              activation_fn=activation_fn,
              normalization_op_params=normalization_op_params)
    x = _conv(x,
              filters,
              1,
              activation_fn=tf.identity,
              normalization_op_params=normalization_op_params)
    if residual:
        x = x + shortcut
    return x


def mobiledet_cpu_backbone(h, multiplier=1.0, normalization_op_params=None):
    """Build a MobileDet CPU backbone."""

    def _scale(filters):
        return _scale_filters(filters, multiplier)

    ibn = functools.partial(_inverted_bottleneck,
                            use_se=True,
                            activation_fn=_swish6,
                            normalization_op_params=normalization_op_params)

    endpoints = {}
    h = _conv(h,
              _scale(16),
              3,
              strides=2,
              activation_fn=_swish6,
              normalization_op_params=normalization_op_params)
    h = _inverted_bottleneck_no_expansion(
        h,
        _scale(8),
        use_se=True,
        activation_fn=_swish6,
        normalization_op_params=normalization_op_params)
    endpoints['1'] = h

    h = ibn(h, _scale(16), expansion=4, strides=2, residual=False)
    endpoints['2'] = h

    h = ibn(h, _scale(32), expansion=8, strides=2, residual=False)
    h = ibn(h, _scale(32), expansion=4)
    h = ibn(h, _scale(32), expansion=4)
    h = ibn(h, _scale(32), expansion=4)
    endpoints['3'] = h

    h = ibn(h,
            _scale(72),
            kernel_size=5,
            expansion=8,
            strides=2,
            residual=False)
    h = ibn(h, _scale(72), expansion=8)
    h = ibn(h, _scale(72), kernel_size=5, expansion=4)
    h = ibn(h, _scale(72), expansion=4)
    h = ibn(h, _scale(72), expansion=8, residual=False)
    h = ibn(h, _scale(72), expansion=8)
    h = ibn(h, _scale(72), expansion=8)
    h = ibn(h, _scale(72), expansion=8)
    endpoints['4'] = h

    h = ibn(h,
            _scale(104),
            kernel_size=5,
            expansion=8,
            strides=2,
            residual=False)
    h = ibn(h, _scale(104), kernel_size=5, expansion=4)
    h = ibn(h, _scale(104), kernel_size=5, expansion=4)
    h = ibn(h, _scale(104), expansion=4)
    h = ibn(h, _scale(144), expansion=8, residual=False)
    endpoints['5'] = h

    return endpoints


def mobiledet_dsp_backbone(h, multiplier=1.0, normalization_op_params=None):
    """Build a MobileDet DSP backbone."""

    def _scale(filters):
        return _scale_filters(filters, multiplier)

    ibn = functools.partial(_inverted_bottleneck,
                            activation_fn=tf.nn.relu6,
                            normalization_op_params=normalization_op_params)
    fused = functools.partial(_fused_conv,
                              activation_fn=tf.nn.relu6,
                              normalization_op_params=normalization_op_params)
    tucker = functools.partial(_tucker_conv,
                               activation_fn=tf.nn.relu6,
                               normalization_op_params=normalization_op_params)

    endpoints = {}
    h = _conv(h,
              _scale(32),
              3,
              strides=2,
              activation_fn=tf.nn.relu6,
              normalization_op_params=normalization_op_params)
    h = _inverted_bottleneck_no_expansion(
        h,
        _scale(24),
        activation_fn=tf.nn.relu6,
        normalization_op_params=normalization_op_params)
    endpoints['1'] = h

    h = fused(h, _scale(32), expansion=4, strides=2, residual=False)
    h = fused(h, _scale(32), expansion=4)
    h = ibn(h, _scale(32), expansion=4)
    h = tucker(h, _scale(32), input_rank_ratio=0.25, output_rank_ratio=0.75)
    endpoints['2'] = h

    h = fused(h, _scale(64), expansion=8, strides=2, residual=False)
    h = ibn(h, _scale(64), expansion=4)
    h = fused(h, _scale(64), expansion=4)
    h = fused(h, _scale(64), expansion=4)
    endpoints['3'] = h

    h = fused(h, _scale(120), expansion=8, strides=2, residual=False)
    h = ibn(h, _scale(120), expansion=4)
    h = ibn(h, _scale(120), expansion=8)
    h = ibn(h, _scale(120), expansion=8)
    h = fused(h, _scale(144), expansion=8, residual=False)
    h = ibn(h, _scale(144), expansion=8)
    h = ibn(h, _scale(144), expansion=8)
    h = ibn(h, _scale(144), expansion=8)
    endpoints['4'] = h

    h = ibn(h, _scale(160), expansion=4, strides=2, residual=False)
    h = ibn(h, _scale(160), expansion=4)
    h = fused(h, _scale(160), expansion=4)
    h = tucker(h, _scale(160), input_rank_ratio=0.75, output_rank_ratio=0.75)
    h = ibn(h, _scale(240), expansion=8, residual=False)
    endpoints['5'] = h

    return endpoints


def mobiledet_edgetpu_backbone(h, multiplier=1.0, normalization_op_params=None):
    """Build a MobileDet EdgeTPU backbone."""

    def _scale(filters):
        return _scale_filters(filters, multiplier)

    ibn = functools.partial(_inverted_bottleneck,
                            activation_fn=tf.nn.relu6,
                            normalization_op_params=normalization_op_params)
    fused = functools.partial(_fused_conv,
                              activation_fn=tf.nn.relu6,
                              normalization_op_params=normalization_op_params)
    tucker = functools.partial(_tucker_conv,
                               activation_fn=tf.nn.relu6,
                               normalization_op_params=normalization_op_params)

    endpoints = {}
    h = _conv(h,
              _scale(32),
              3,
              strides=2,
              activation_fn=tf.nn.relu6,
              normalization_op_params=normalization_op_params)
    h = tucker(h,
               _scale(16),
               input_rank_ratio=0.25,
               output_rank_ratio=0.75,
               residual=False)
    endpoints['1'] = h

    h = fused(h, _scale(16), expansion=8, strides=2, residual=False)
    h = fused(h, _scale(16), expansion=4)
    h = fused(h, _scale(16), expansion=8)
    h = fused(h, _scale(16), expansion=4)
    endpoints['2'] = h

    h = fused(h,
              _scale(40),
              expansion=8,
              kernel_size=5,
              strides=2,
              residual=False)
    h = fused(h, _scale(40), expansion=4)
    h = fused(h, _scale(40), expansion=4)
    h = fused(h, _scale(40), expansion=4)
    endpoints['3'] = h

    h = ibn(h, _scale(72), expansion=8, strides=2, residual=False)
    h = ibn(h, _scale(72), expansion=8)
    h = fused(h, _scale(72), expansion=4)
    h = fused(h, _scale(72), expansion=4)
    h = ibn(h, _scale(96), expansion=8, kernel_size=5, residual=False)
    h = ibn(h, _scale(96), expansion=8, kernel_size=5)
    h = ibn(h, _scale(96), expansion=8)
    h = ibn(h, _scale(96), expansion=8)
    endpoints['4'] = h

    h = ibn(h,
            _scale(120),
            expansion=8,
            kernel_size=5,
            strides=2,
            residual=False)
    h = ibn(h, _scale(120), expansion=8)
    h = ibn(h, _scale(120), expansion=4, kernel_size=5)
    h = ibn(h, _scale(120), expansion=8)
    h = ibn(h, _scale(384), expansion=8, kernel_size=5, residual=False)
    endpoints['5'] = h

    return endpoints


def mobiledet_gpu_backbone(h, multiplier=1.0, normalization_op_params=None):
    """Build a MobileDet GPU backbone."""

    def _scale(filters):
        return _scale_filters(filters, multiplier)

    ibn = functools.partial(_inverted_bottleneck,
                            activation_fn=tf.nn.relu6,
                            normalization_op_params=normalization_op_params)
    fused = functools.partial(_fused_conv,
                              activation_fn=tf.nn.relu6,
                              normalization_op_params=normalization_op_params)
    tucker = functools.partial(_tucker_conv,
                               activation_fn=tf.nn.relu6,
                               normalization_op_params=normalization_op_params)

    endpoints = {}
    # block 0
    h = _conv(h,
              _scale(32),
              3,
              strides=2,
              activation_fn=tf.nn.relu6,
              normalization_op_params=normalization_op_params)

    # block 1
    h = tucker(h,
               _scale(16),
               input_rank_ratio=0.25,
               output_rank_ratio=0.25,
               residual=False)
    endpoints['1'] = h

    # block 2
    h = fused(h, _scale(32), expansion=8, strides=2, residual=False)
    h = tucker(h, _scale(32), input_rank_ratio=0.25, output_rank_ratio=0.25)
    h = tucker(h, _scale(32), input_rank_ratio=0.25, output_rank_ratio=0.25)
    h = tucker(h, _scale(32), input_rank_ratio=0.25, output_rank_ratio=0.25)
    endpoints['2'] = h

    # block 3
    h = fused(h,
              _scale(64),
              expansion=8,
              kernel_size=3,
              strides=2,
              residual=False)
    h = fused(h, _scale(64), expansion=8)
    h = fused(h, _scale(64), expansion=8)
    h = fused(h, _scale(64), expansion=4)
    endpoints['3'] = h

    # block 4
    h = fused(h,
              _scale(128),
              expansion=8,
              kernel_size=3,
              strides=2,
              residual=False)
    h = fused(h, _scale(128), expansion=4)
    h = fused(h, _scale(128), expansion=4)
    h = fused(h, _scale(128), expansion=4)

    # block 5
    h = fused(h,
              _scale(128),
              expansion=8,
              kernel_size=3,
              strides=1,
              residual=False)
    h = fused(h, _scale(128), expansion=8)
    h = fused(h, _scale(128), expansion=8)
    h = fused(h, _scale(128), expansion=8)
    endpoints['4'] = h

    # block 6
    h = fused(h,
              _scale(128),
              expansion=4,
              kernel_size=3,
              strides=2,
              residual=False)
    h = fused(h, _scale(128), expansion=4)
    h = fused(h, _scale(128), expansion=4)
    h = fused(h, _scale(128), expansion=4)

    # block 7
    h = ibn(h,
            _scale(384),
            expansion=8,
            kernel_size=3,
            strides=1,
            residual=False)
    endpoints['5'] = h

    return endpoints


class MobileDet(tf.keras.Model):
    _MODEL_FN = {
        'mobiledet_cpu': mobiledet_cpu_backbone,
        'mobiledet_gpu': mobiledet_gpu_backbone,
        'mobiledet_edge_tpu': mobiledet_edgetpu_backbone,
        'mobiledet_dsp': mobiledet_dsp_backbone
    }

    def __init__(
            self,
            input_shape,
            model_name=None,
            multiplier=1.0,
            checkpoint=None,
            normalization_op_params=None):

        input_layer = tf.keras.Input(shape=input_shape, name="resnet_input")
        outputs = MobileDet._MODEL_FN[model_name](
            input_layer,
            multiplier=multiplier,
            normalization_op_params=normalization_op_params)

        super(MobileDet, self).__init__(
            inputs=[input_layer],
            outputs=outputs,
            name=model_name)

        if checkpoint:
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint)
            self.load_weights(latest_checkpoint).assert_consumed()
            logging.info(
                'Initialized weights from {}'.format(latest_checkpoint))
        else:
            logging.warning('Proceeding with random initialization!')
