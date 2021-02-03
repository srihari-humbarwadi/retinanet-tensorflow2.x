# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Contains definitions for the post-activation form of Residual Networks.
Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import functools

import tensorflow as tf
from absl import logging

from retinanet.model.utils import get_normalization_op


class NormActivation:
    """Combined Normalization and Activation layers."""

    def __init__(self,
                 momentum=0.997,
                 init_zero=False,
                 use_activation=True,
                 activation='relu',
                 fused=True,
                 name=None):
        """A class to construct layers for a batch normalization followed by a ReLU.
    Args:
      momentum: momentum for the moving average.
      epsilon: small float added to variance to avoid dividing by zero.
      init_zero: `bool` if True, initializes scale parameter of batch
          normalization with 0. If False, initialize it with 1.
      fused: `bool` fused option in batch normalziation.
      use_actiation: `bool`, whether to add the optional activation layer after
        the batch normalization layer.
      activation: 'string', the type of the activation layer. Currently support
        `relu` and `swish`.
      name: `str` name for the operation.
    """
        if init_zero:
            gamma_initializer = tf.keras.initializers.Zeros()
        else:
            gamma_initializer = tf.keras.initializers.Ones()

        normalization_op = get_normalization_op()

        self._normalization_op = normalization_op(
            momentum=momentum,
            epsilon=1e-4,
            center=True,
            scale=True,
            gamma_initializer=gamma_initializer,
            name=name)
        self._use_activation = use_activation
        if activation == 'relu':
            self._activation_op = tf.nn.relu
        elif activation == 'swish':
            self._activation_op = tf.nn.swish
        else:
            raise ValueError('Unsupported activation `{}`.'.format(activation))

    def __call__(self, inputs):
        inputs = self._normalization_op(inputs)

        if self._use_activation:
            inputs = self._activation_op(inputs)
        return inputs


def norm_activation_builder(momentum=0.997,
                            activation='relu',
                            **kwargs):
    return functools.partial(NormActivation,
                             momentum=momentum,
                             activation=activation,
                             **kwargs)


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
      inputs: `Tensor` of size `[batch, channels, height, width]` or
          `[batch, height, width, channels]` depending on `data_format`.
      kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
          operations. Should be a positive integer.
    Returns:
      A padded `Tensor` of the same `data_format` with size either intact
      (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'channels_first':
        padded_inputs = tf.pad(tensor=inputs,
                               paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                         [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(tensor=inputs,
                               paddings=[[0, 0], [pad_beg, pad_end],
                                         [pad_beg, pad_end], [0, 0]])

    return padded_inputs


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format='channels_last'):
    """Strided 2-D convolution with explicit padding.
    The padding is consistent and is based only on `kernel_size`, not on the
    dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    Args:
      inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
      filters: `int` number of filters in the convolution.
      kernel_size: `int` size of the kernel to be used in the convolution.
      strides: `int` strides of the convolution.
    Returns:
      A `Tensor` of shape `[batch, filters, height_out, width_out]`.
    """
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.initializers.VarianceScaling(),
        data_format=data_format)(inputs=inputs)


def residual_block(inputs,
                   filters,
                   strides,
                   activation_op=tf.nn.relu,
                   norm_activation=norm_activation_builder(activation='relu'),
                   use_projection=False):
    """Standard building block for residual networks with BN after convolutions.
    Args:
      inputs: `Tensor` of size `[batch, channels, height, width]`.
      filters: `int` number of filters for the first two convolutions. Note that
          the third and final convolution will use 4 times as many filters.
      strides: `int` block stride. If greater than 1, this block will ultimately
          downsample the input.
      use_projection: `bool` for whether this block should use a projection
          shortcut (versus the default identity shortcut). This is usually
          `True` for the first block of a block group, which may change the
          number of filters and the resolution.
    Returns:
      The output `Tensor` of the block.
    """
    shortcut = inputs
    if use_projection:
        # Projection shortcut in first layer to match filters and strides
        shortcut = conv2d_fixed_padding(inputs=inputs,
                                        filters=filters,
                                        kernel_size=1,
                                        strides=strides)
        shortcut = norm_activation(use_activation=False)(shortcut)

    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=filters,
                                  kernel_size=3,
                                  strides=strides)
    inputs = norm_activation()(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=filters,
                                  kernel_size=3,
                                  strides=1)
    inputs = norm_activation(use_activation=False, init_zero=True)(inputs)

    return activation_op(inputs + shortcut)


def bottleneck_block(
        inputs,
        filters,
        strides,
        activation_op=tf.nn.relu,
        norm_activation=norm_activation_builder(activation='relu'),
        use_projection=False):
    """Bottleneck block variant for residual networks with BN after convolutions.
    Args:
      inputs: `Tensor` of size `[batch, channels, height, width]`.
      filters: `int` number of filters for the first two convolutions. Note that
          the third and final convolution will use 4 times as many filters.
      strides: `int` block stride. If greater than 1, this block will ultimately
          downsample the input.
      use_projection: `bool` for whether this block should use a projection
          shortcut (versus the default identity shortcut). This is usually
          `True` for the first block of a block group, which may change the
          number of filters and the resolution.
    Returns:
      The output `Tensor` of the block.
    """
    shortcut = inputs
    if use_projection:
        # Projection shortcut only in first block within a group. Bottleneck
        # blocks end with 4 times the number of filters.
        filters_out = 4 * filters
        shortcut = conv2d_fixed_padding(inputs=inputs,
                                        filters=filters_out,
                                        kernel_size=1,
                                        strides=strides)
        shortcut = norm_activation(use_activation=False)(shortcut)

    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=filters,
                                  kernel_size=1,
                                  strides=1)
    inputs = norm_activation()(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=filters,
                                  kernel_size=3,
                                  strides=strides)
    inputs = norm_activation()(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=4 * filters,
                                  kernel_size=1,
                                  strides=1)
    inputs = norm_activation(use_activation=False, init_zero=True)(inputs)

    return activation_op(inputs + shortcut)


def block_group(inputs, filters, block_fn, blocks, strides, name):
    """Creates one group of blocks for the ResNet model.
    Args:
      inputs: `Tensor` of size `[batch, channels, height, width]`.
      filters: `int` number of filters for the first convolution of the layer.
      block_fn: `function` for the block to use within the model
      blocks: `int` number of blocks contained in the layer.
      strides: `int` stride to use for the first convolution of the layer. If
          greater than 1, this layer will downsample the input.
      name: `str`name for the Tensor output of the block layer.
    Returns:
      The output `Tensor` of the block layer.
    """
    # Only the first block per block_group uses projection shortcut and
    # strides.
    inputs = block_fn(inputs, filters, strides, use_projection=True)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, 1)

    return tf.identity(inputs, name)


def resnet_fn(input_layer,
              block_fn,
              layers,
              norm_activation=norm_activation_builder(activation='relu')):
    inputs = conv2d_fixed_padding(inputs=input_layer,
                                  filters=64,
                                  kernel_size=7,
                                  strides=2)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = norm_activation()(inputs)

    inputs = tf.keras.layers.MaxPool2D(pool_size=3,
                                       strides=2,
                                       padding='SAME',
                                       name='initial_max_pool')(inputs)
    c2 = block_group(inputs=inputs,
                     filters=64,
                     block_fn=block_fn,
                     blocks=layers[0],
                     strides=1,
                     name='block_group1')
    c3 = block_group(inputs=c2,
                     filters=128,
                     block_fn=block_fn,
                     blocks=layers[1],
                     strides=2,
                     name='block_group2')
    c4 = block_group(inputs=c3,
                     filters=256,
                     block_fn=block_fn,
                     blocks=layers[2],
                     strides=2,
                     name='block_group3')
    c5 = block_group(inputs=c4,
                     filters=512,
                     block_fn=block_fn,
                     blocks=layers[3],
                     strides=2,
                     name='block_group4')
    return {
        '2': c2,
        '3': c3,
        '4': c4,
        '5': c5
    }


class ResNet(tf.keras.Model):
    _MODEL_CONFIG = {
        10: {
            'block': residual_block,
            'layers': [1, 1, 1, 1]
        },
        18: {
            'block': residual_block,
            'layers': [2, 2, 2, 2]
        },
        34: {
            'block': residual_block,
            'layers': [3, 4, 6, 3]
        },
        50: {
            'block': bottleneck_block,
            'layers': [3, 4, 6, 3]
        },
        101: {
            'block': bottleneck_block,
            'layers': [3, 4, 23, 3]
        },
        152: {
            'block': bottleneck_block,
            'layers': [3, 8, 36, 3]
        },
        200: {
            'block': bottleneck_block,
            'layers': [3, 24, 36, 3]
        }
    }

    def __init__(self, input_shape, depth, checkpoint=None):
        input_layer = tf.keras.Input(shape=input_shape, name="resnet_input")
        outputs = resnet_fn(
            input_layer,
            block_fn=ResNet._MODEL_CONFIG[depth]['block'],
            layers=ResNet._MODEL_CONFIG[depth]['layers'])

        super(ResNet, self).__init__(
            inputs=[input_layer],
            outputs=outputs,
            name='resnet_' + str(depth))

        if checkpoint:
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint)
            self.load_weights(latest_checkpoint).assert_consumed()
            logging.info(
                'Initialized weights from {}'.format(latest_checkpoint))
        else:
            logging.warning('Proceeding with random initialization!')
