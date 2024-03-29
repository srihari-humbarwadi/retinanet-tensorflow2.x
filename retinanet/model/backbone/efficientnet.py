# Copyright 2020 Google Research. All Rights Reserved.
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
import collections
import itertools
import math
import re

import numpy as np
import six
import tensorflow as tf
from absl import logging

from retinanet.model.utils import get_normalization_op


GlobalParams = collections.namedtuple('GlobalParams', [
    'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'survival_prob', 'relu_fn', 'batch_norm', 'use_se',
    'local_pooling', 'condconv_num_experts', 'clip_projection_output',
    'blocks_args', 'fix_head_stem', 'grad_checkpoint'
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type', 'fused_conv',
    'super_pixel', 'condconv'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def efficientnet_params(model_name):
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


def efficientnet_lite_params(model_name):
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
        'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
        'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
        'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
        'efficientnet-lite4': (1.4, 1.8, 300, 0.3),
    }
    return params_dict[model_name]


_EFFICIENTNET_DEFAULT_BLOCKS_ARGS = [
    'r1_k3_s11_e1_i32_o16_se0.25',
    'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25',
    'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25',
    'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25',
]

_EFFICIENTNET_LITE_DEFAULT_BLOCKS_ARGS = [
    'r1_k3_s11_e1_i32_o16_se0.25',
    'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25',
    'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25',
    'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25',
]


def drop_connect(inputs, is_training, survival_prob, seed=0):
    """Drop the entire conv with given survival probability."""
    # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    if not is_training:
        return inputs

    # Compute tensor.
    batch_size = tf.shape(inputs)[0]
    random_tensor = survival_prob
    random_tensor += tf.random.uniform(
      [batch_size, 1, 1, 1], dtype=inputs.dtype, seed=seed)
    binary_tensor = tf.floor(random_tensor)
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no addition compute is
    # needed at test time.
    output = inputs / survival_prob * binary_tensor
    return output


def conv_kernel_initializer(shape, dtype=None, partition_info=None, seed=0):
    """Initialization for convolutional kernels.

  The main difference with tf.variance_scaling_initializer is that
  tf.variance_scaling_initializer uses a truncated normal with an uncorrected
  standard deviation, whereas here we use a normal distribution. Similarly,
  tf.initializers.variance_scaling uses a truncated normal with
  a corrected standard deviation.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random.normal(shape,
                            mean=0.0,
                            stddev=np.sqrt(2.0 / fan_out),
                            dtype=dtype,
                            seed=seed)


def dense_kernel_initializer(shape, dtype=None, partition_info=None, seed=0):
    """Initialization for dense kernels.

  This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                    distribution='uniform').
  It is written out explicitly here for clarity.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
    del partition_info
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random.uniform(
      shape, -init_range, init_range, dtype=dtype, seed=seed)


def superpixel_kernel_initializer(shape, dtype='float32', partition_info=None):
    """Initializes superpixel kernels.

  This is inspired by space-to-depth transformation that is mathematically
  equivalent before and after the transformation. But we do the space-to-depth
  via a convolution. Moreover, we make the layer trainable instead of direct
  transform, we can initialization it this way so that the model can learn not
  to do anything but keep it mathematically equivalent, when improving
  performance.


  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
    del partition_info
    #  use input depth to make superpixel kernel.
    depth = shape[-2]
    filters = np.zeros([2, 2, depth, 4 * depth], dtype=dtype)
    i = np.arange(2)
    j = np.arange(2)
    k = np.arange(depth)
    mesh = np.array(np.meshgrid(i, j, k)).T.reshape(-1, 3).T
    filters[mesh[0], mesh[1], mesh[2], 4 * mesh[2] + 2 * mesh[0] + mesh[1]] = 1
    return filters


def round_filters(filters, global_params, skip=False):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if skip or not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth,
                      int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params, skip=False):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if skip or not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class SE(tf.keras.layers.Layer):
    """Squeeze-and-excitation layer."""

    def __init__(self, global_params, se_filters, output_filters, name=None):
        super().__init__(name=name)

        self._local_pooling = global_params.local_pooling
        self._data_format = global_params.data_format
        self._relu_fn = global_params.relu_fn or tf.nn.swish

        # Squeeze and Excitation layer.
        self._se_reduce = tf.keras.layers.Conv2D(
            se_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=True,
            name='conv2d')
        self._se_expand = tf.keras.layers.Conv2D(
            output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=True,
            name='conv2d_1')

    def call(self, inputs):
        h_axis, w_axis = [2, 3
                          ] if self._data_format == 'channels_first' else [1, 2]
        if self._local_pooling:
            se_tensor = tf.nn.avg_pool(
                inputs,
                ksize=[1, inputs.shape[h_axis], inputs.shape[w_axis], 1],
                strides=[1, 1, 1, 1],
                padding='VALID')
        else:
            se_tensor = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
        se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
        # logging.info('Built SE %s : %s', self.name, se_tensor.shape)
        return tf.sigmoid(se_tensor) * inputs


class SuperPixel(tf.keras.layers.Layer):
    """Super pixel layer."""

    def __init__(self, block_args, global_params, name=None):
        super().__init__(name=name)
        self._superpixel = tf.keras.layers.Conv2D(
            block_args.input_filters,
            kernel_size=[2, 2],
            strides=[2, 2],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=global_params.data_format,
            use_bias=False,
            name='conv2d')
        self._bnsp = global_params.batch_norm(
            axis=1 if global_params.data_format == 'channels_first' else -1,
            name='tpu_batch_normalization')
        self._relu_fn = global_params.relu_fn or tf.nn.swish

    def call(self, inputs, training):
        return self._relu_fn(self._bnsp(self._superpixel(inputs), training))


class MBConvBlock(tf.keras.layers.Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.

  Attributes:
    endpoints: dict. A list of internal tensors.
  """

    def __init__(self, block_args, global_params, name=None):
        """Initializes a MBConv block.

    Args:
      block_args: BlockArgs, arguments to create a Block.
      global_params: GlobalParams, a set of global parameters.
      name: layer name.
    """
        super().__init__(name=name)

        self._block_args = block_args
        self._global_params = global_params
        self._local_pooling = global_params.local_pooling
        self._batch_norm = global_params.batch_norm
        self._condconv_num_experts = global_params.condconv_num_experts
        self._data_format = global_params.data_format
        self._channel_axis = 1 if self._data_format == 'channels_first' else -1

        self._relu_fn = global_params.relu_fn or tf.nn.swish
        self._has_se = (global_params.use_se and
                        self._block_args.se_ratio is not None and
                        0 < self._block_args.se_ratio <= 1)

        self._clip_projection_output = global_params.clip_projection_output

        self.endpoints = None

        if self._block_args.condconv:
            raise ValueError('Condconv is not supported.')

        # Builds the block accordings to arguments.
        self._build()

    @property
    def block_args(self):
        return self._block_args

    def _build(self):
        """Builds block according to the arguments."""
        # pylint: disable=g-long-lambda
        bid = itertools.count(0)

        def get_bn_name(): return 'tpu_batch_normalization' + ('' if not next(
            bid) else '_' + str(next(bid) // 2))
        cid = itertools.count(0)
        def get_conv_name(): return 'conv2d' + ('' if not next(cid) else '_' + str(
            next(cid) // 2))
        # pylint: enable=g-long-lambda

        if self._block_args.super_pixel == 1:
            self.super_pixel = SuperPixel(self._block_args,
                                          self._global_params,
                                          name='super_pixel')
        else:
            self.super_pixel = None

        filters = self._block_args.input_filters * self._block_args.expand_ratio
        kernel_size = self._block_args.kernel_size

        if self._block_args.fused_conv:
            # Fused expansion phase. Called if using fused convolutions.
            self._fused_conv = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=[kernel_size, kernel_size],
                strides=self._block_args.strides,
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=False,
                name=get_conv_name())
        else:
            # Expansion phase. Called if not using fused convolutions and expansion
            # phase is necessary.
            if self._block_args.expand_ratio != 1:
                self._expand_conv = tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=[1, 1],
                    strides=[1, 1],
                    kernel_initializer=conv_kernel_initializer,
                    padding='same',
                    data_format=self._data_format,
                    use_bias=False,
                    name=get_conv_name())
                self._bn0 = self._batch_norm(axis=self._channel_axis,
                                             name=get_bn_name())

            # Depth-wise convolution phase. Called if not using fused convolutions.
            self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=[kernel_size, kernel_size],
                strides=self._block_args.strides,
                depthwise_initializer=conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=False,
                name='depthwise_conv2d')

        self._bn1 = self._batch_norm(axis=self._channel_axis,
                                     name=get_bn_name())

        if self._has_se:
            num_reduced_filters = max(
                1,
                int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se = SE(self._global_params,
                          num_reduced_filters,
                          filters,
                          name='se')
        else:
            self._se = None

        # Output phase.
        filters = self._block_args.output_filters
        self._project_conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False,
            name=get_conv_name())
        self._bn2 = self._batch_norm(axis=self._channel_axis,
                                     name=get_bn_name())

    def call(self, inputs, training, survival_prob=None):
        """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """

        def _call(inputs):
            # logging.info('Block %s input shape: %s', self.name, inputs.shape)
            x = inputs

            # creates conv 2x2 kernel
            if self.super_pixel:
                x = self.super_pixel(x, training)
                # logging.info('SuperPixel %s: %s', self.name, x.shape)

            if self._block_args.fused_conv:
                # If use fused mbconv, skip expansion and use regular conv.
                x = self._relu_fn(
                    self._bn1(self._fused_conv(x), training=training))
                # logging.info('Conv2D shape: %s', x.shape)
            else:
                # Otherwise, first apply expansion and then apply depthwise conv.
                if self._block_args.expand_ratio != 1:
                    x = self._relu_fn(
                        self._bn0(self._expand_conv(x), training=training))
                    # logging.info('Expand shape: %s', x.shape)

                x = self._relu_fn(
                    self._bn1(self._depthwise_conv(x), training=training))
                # logging.info('DWConv shape: %s', x.shape)

            if self._se:
                x = self._se(x)

            self.endpoints = {'expansion_output': x}

            x = self._bn2(self._project_conv(x), training=training)
            # Add identity so that quantization-aware training can insert
            # quantization ops correctly.
            x = tf.identity(x)
            if self._clip_projection_output:
                x = tf.clip_by_value(x, -6, 6)
            if self._block_args.id_skip:
                if all(
                        s == 1 for s in self._block_args.strides
                ) and self._block_args.input_filters == \
                        self._block_args.output_filters:
                    # Apply only if skip connection presents.
                    if survival_prob:
                        x = drop_connect(x, training, survival_prob)
                    x = tf.add(x, inputs)
            # logging.info('Project shape: %s', x.shape)
            return x

        return _call(inputs)


class MBConvBlockWithoutDepthwise(MBConvBlock):
    """MBConv-like block without depthwise convolution and squeeze-and-excite."""

    def _build(self):
        """Builds block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        # pylint: disable=g-long-lambda
        cid = itertools.count(0)
        def get_conv_name(): return 'conv2d' + ('' if not next(cid) else '_' + str(
            next(cid) // 2))
        # pylint: enable=g-long-lambda
        kernel_size = self._block_args.kernel_size
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tf.keras.layers.Conv2D(
                filters,
                kernel_size=[kernel_size, kernel_size],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=False,
                name=get_conv_name())
            self._bn0 = self._batch_norm(axis=self._channel_axis)

        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=self._block_args.strides,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False,
            name=get_conv_name())
        self._bn1 = self._batch_norm(axis=self._channel_axis)

    def call(self, inputs, training, survival_prob=None):
        """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """

        def _call(inputs):
            # logging.info('Block %s  input shape: %s', self.name, inputs.shape)
            if self._block_args.expand_ratio != 1:
                x = self._relu_fn(
                    self._bn0(self._expand_conv(inputs), training=training))
            else:
                x = inputs
            # logging.info('Expand shape: %s', x.shape)

            self.endpoints = {'expansion_output': x}

            x = self._bn1(self._project_conv(x), training=training)
            # Add identity so that quantization-aware training can insert
            # quantization ops correctly.
            x = tf.identity(x)
            if self._clip_projection_output:
                x = tf.clip_by_value(x, -6, 6)

            if self._block_args.id_skip:
                if all(
                        s == 1 for s in self._block_args.strides
                ) and self._block_args.input_filters == \
                        self._block_args.output_filters:
                    # Apply only if skip connection presents.
                    if survival_prob:
                        x = drop_connect(x, training, survival_prob)
                    x = tf.add(x, inputs)
            # logging.info('Project shape: %s', x.shape)
            return x

        return _call(inputs)


class Stem(tf.keras.layers.Layer):
    """Stem layer at the begining of the network."""

    def __init__(self, global_params, stem_filters, name=None):
        super().__init__(name=name)
        self._conv_stem = tf.keras.layers.Conv2D(
            filters=round_filters(stem_filters, global_params,
                                  global_params.fix_head_stem),
            kernel_size=[3, 3],
            strides=[2, 2],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=global_params.data_format,
            use_bias=False)
        self._bn = global_params.batch_norm(
            axis=(1 if global_params.data_format == 'channels_first' else -1))
        self._relu_fn = global_params.relu_fn or tf.nn.swish

    def call(self, inputs, training):
        return self._relu_fn(
            self._bn(self._conv_stem(inputs), training=training))


class Head(tf.keras.layers.Layer):
    """Head layer for network outputs."""

    def __init__(self, global_params, name=None):
        super().__init__(name=name)

        self.endpoints = {}
        self._global_params = global_params

        self._conv_head = tf.keras.layers.Conv2D(
            filters=round_filters(1280, global_params,
                                  global_params.fix_head_stem),
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=global_params.data_format,
            use_bias=False,
            name='conv2d')
        self._bn = global_params.batch_norm(
            axis=(1 if global_params.data_format == 'channels_first' else -1))
        self._relu_fn = global_params.relu_fn or tf.nn.swish

        self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
            data_format=global_params.data_format)
        if global_params.num_classes:
            self._fc = tf.keras.layers.Dense(
                global_params.num_classes,
                kernel_initializer=dense_kernel_initializer)
        else:
            self._fc = None

        if global_params.dropout_rate > 0:
            self._dropout = tf.keras.layers.Dropout(global_params.dropout_rate)
        else:
            self._dropout = None

        self.h_axis, self.w_axis = ([2, 3] if global_params.data_format
                                    == 'channels_first' else [1, 2])

    def call(self, inputs, training, pooled_features_only):
        """Call the layer."""
        outputs = self._relu_fn(
            self._bn(self._conv_head(inputs), training=training))
        self.endpoints['head_1x1'] = outputs

        if self._global_params.local_pooling:
            shape = outputs.get_shape().as_list()
            kernel_size = [1, shape[self.h_axis], shape[self.w_axis], 1]
            outputs = tf.nn.avg_pool(outputs,
                                     ksize=kernel_size,
                                     strides=[1, 1, 1, 1],
                                     padding='VALID')
            self.endpoints['pooled_features'] = outputs
            if not pooled_features_only:
                if self._dropout:
                    outputs = self._dropout(outputs, training=training)
                self.endpoints['global_pool'] = outputs
                if self._fc:
                    outputs = tf.squeeze(outputs, [self.h_axis, self.w_axis])
                    outputs = self._fc(outputs)
                self.endpoints['head'] = outputs
        else:
            outputs = self._avg_pooling(outputs)
            self.endpoints['pooled_features'] = outputs
            if not pooled_features_only:
                if self._dropout:
                    outputs = self._dropout(outputs, training=training)
                self.endpoints['global_pool'] = outputs
                if self._fc:
                    outputs = self._fc(outputs)
                self.endpoints['head'] = outputs
        return outputs


class Model(tf.keras.Model):
    """A class implements tf.keras.Model.

    Reference: https://arxiv.org/abs/1807.11626
  """

    def __init__(self, blocks_args=None, global_params=None, name=None):
        """Initializes an `Model` instance.

    Args:
      blocks_args: A list of BlockArgs to construct block modules.
      global_params: GlobalParams, a set of global parameters.
      name: A string of layer name.

    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
        super().__init__(name=name)

        if not isinstance(blocks_args, list):
            raise ValueError('blocks_args should be a list.')
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._relu_fn = global_params.relu_fn or tf.nn.swish
        self._batch_norm = global_params.batch_norm
        self._fix_head_stem = global_params.fix_head_stem

        self.endpoints = None

        self._build()

    def _get_conv_block(self, conv_type):
        conv_block_map = {0: MBConvBlock, 1: MBConvBlockWithoutDepthwise}
        return conv_block_map[conv_type]

    def _build(self):
        """Builds a model."""
        self._blocks = []

        # Stem part.
        self._stem = Stem(self._global_params,
                          self._blocks_args[0].input_filters)

        # Builds blocks.
        block_id = itertools.count(0)
        def block_name(): return 'blocks_%d' % next(block_id)
        for i, block_args in enumerate(self._blocks_args):
            assert block_args.num_repeat > 0
            assert block_args.super_pixel in [0, 1, 2]
            # Update block input and output filters based on depth multiplier.
            input_filters = round_filters(block_args.input_filters,
                                          self._global_params)

            output_filters = round_filters(block_args.output_filters,
                                           self._global_params)
            kernel_size = block_args.kernel_size
            if self._fix_head_stem and (i == 0 or
                                        i == len(self._blocks_args) - 1):
                repeats = block_args.num_repeat
            else:
                repeats = round_repeats(block_args.num_repeat,
                                        self._global_params)
            block_args = block_args._replace(input_filters=input_filters,
                                             output_filters=output_filters,
                                             num_repeat=repeats)

            # The first block needs to take care of stride and filter size increase.
            conv_block = self._get_conv_block(block_args.conv_type)
            if not block_args.super_pixel:  # no super_pixel at all
                self._blocks.append(
                    conv_block(block_args,
                               self._global_params,
                               name=block_name()))
            else:
                # if superpixel, adjust filters, kernels, and strides.
                depth_factor = int(4 / block_args.strides[0] /
                                   block_args.strides[1])
                block_args = block_args._replace(
                    input_filters=block_args.input_filters * depth_factor,
                    output_filters=block_args.output_filters * depth_factor,
                    kernel_size=((block_args.kernel_size + 1) // 2 if
                                 depth_factor > 1 else block_args.kernel_size))
                # if the first block has stride-2 and super_pixel trandformation
                if (block_args.strides[0] == 2 and block_args.strides[1] == 2):
                    block_args = block_args._replace(strides=[1, 1])
                    self._blocks.append(
                        conv_block(block_args,
                                   self._global_params,
                                   name=block_name()))
                    block_args = block_args._replace(  # sp stops at stride-2
                        super_pixel=0,
                        input_filters=input_filters,
                        output_filters=output_filters,
                        kernel_size=kernel_size)
                elif block_args.super_pixel == 1:
                    self._blocks.append(
                        conv_block(block_args,
                                   self._global_params,
                                   name=block_name()))
                    block_args = block_args._replace(super_pixel=2)
                else:
                    self._blocks.append(
                        conv_block(block_args,
                                   self._global_params,
                                   name=block_name()))
            if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                # pylint: enable=protected-access
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    conv_block(block_args,
                               self._global_params,
                               name=block_name()))

        # Head part.
        self._head = Head(self._global_params)

    def call(self,
             inputs,
             training,
             features_only=None,
             pooled_features_only=False):
        """Implementation of call().

    Args:
      inputs: input tensors.
      training: boolean, whether the model is constructed for training.
      features_only: build the base feature network only.
      pooled_features_only: build the base network for features extraction
        (after 1x1 conv layer and global pooling, but before dropout and fc
        head).

    Returns:
      output tensors.
    """
        outputs = None
        self.endpoints = {}
        reduction_idx = 0

        # Calls Stem layers
        outputs = self._stem(inputs, training)
        # logging.info('Built stem %s : %s', self._stem.name, outputs.shape)
        self.endpoints['stem'] = outputs

        # Calls blocks.
        for idx, block in enumerate(self._blocks):
            is_reduction = False  # reduction flag for blocks after the stem layer
            # If the first block has super-pixel (space-to-depth) layer, then stem is
            # the first reduction point.
            if (block.block_args.super_pixel == 1 and idx == 0):
                reduction_idx += 1
                self.endpoints['reduction_%s' % reduction_idx] = outputs

            elif ((idx == len(self._blocks) - 1) or
                  self._blocks[idx + 1].block_args.strides[0] > 1):
                is_reduction = True
                reduction_idx += 1

            survival_prob = self._global_params.survival_prob
            if survival_prob:
                drop_rate = 1.0 - survival_prob
                survival_prob = 1.0 - drop_rate * float(idx) / len(self._blocks)
                # logging.info('block_%s survival_prob: %s', idx, survival_prob)
            outputs = block(outputs,
                            training=training,
                            survival_prob=survival_prob)
            self.endpoints['block_%s' % idx] = outputs
            if is_reduction:
                self.endpoints['reduction_%s' % reduction_idx] = outputs
            if block.endpoints:
                for k, v in six.iteritems(block.endpoints):
                    self.endpoints['block_%s/%s' % (idx, k)] = v
                    if is_reduction:
                        self.endpoints['reduction_%s/%s' %
                                       (reduction_idx, k)] = v
        self.endpoints['features'] = outputs

        if not features_only:
            # Calls final layers and returns logits.
            outputs = self._head(outputs, training, pooled_features_only)
            self.endpoints.update(self._head.endpoints)

        return [outputs] + list(
            filter(lambda endpoint: endpoint is not None, [
                self.endpoints.get('reduction_1'),
                self.endpoints.get('reduction_2'),
                self.endpoints.get('reduction_3'),
                self.endpoints.get('reduction_4'),
                self.endpoints.get('reduction_5'),
            ]))


class BlockDecoder(object):
    """Block Decoder for readability."""

    def _decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=[int(options['s'][0]),
                     int(options['s'][1])],
            conv_type=int(options['c']) if 'c' in options else 0,
            fused_conv=int(options['f']) if 'f' in options else 0,
            super_pixel=int(options['p']) if 'p' in options else 0,
            condconv=('cc' in block_string))

    def _encode_block_string(self, block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters,
            'c%d' % block.conv_type,
            'f%d' % block.fused_conv,
            'p%d' % block.super_pixel,
        ]
        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:  # pylint: disable=g-bool-id-comparison
            args.append('noskip')
        if block.condconv:
            args.append('cc')
        return '_'.join(args)

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.
        Args:
          string_list: a list of strings, each string is a notation of block.
        Returns:
          A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        """Encodes a list of Blocks to a list of strings.
        Args:
          blocks_args: A list of namedtuples to represent blocks arguments.
        Returns:
          a list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None,
                 depth_coefficient=None,
                 dropout_rate=0.2,
                 survival_prob=0.8,
                 normalization_op_params=None):
    """Creates a efficientnet model."""
    global_params = GlobalParams(
        blocks_args=_EFFICIENTNET_DEFAULT_BLOCKS_ARGS,
        dropout_rate=dropout_rate,
        survival_prob=survival_prob,
        data_format='channels_last',
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        relu_fn=tf.nn.swish,
        batch_norm=get_normalization_op(**normalization_op_params),
        use_se=True,
        clip_projection_output=False)
    return global_params


def efficientnet_lite(width_coefficient=None,
                      depth_coefficient=None,
                      dropout_rate=0.2,
                      survival_prob=0.8,
                      normalization_op_params=None):
    """Creates a efficientnet model."""
    global_params = GlobalParams(
        blocks_args=_EFFICIENTNET_LITE_DEFAULT_BLOCKS_ARGS,
        dropout_rate=dropout_rate,
        survival_prob=survival_prob,
        data_format='channels_last',
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        relu_fn=tf.nn.relu6,  # Relu6 is for easier quantization.
        # TPU-specific requirement.
        batch_norm=get_normalization_op(**normalization_op_params),
        clip_projection_output=False,
        fix_head_stem=True,  # Don't scale stem and head.
        local_pooling=True,  # special cases for tflite issues.
        use_se=False)  # SE is not well supported on many lite devices.
    return global_params


def get_model_params(model_name, override_params, normalization_op_params):
    if model_name.startswith('efficientnet-b'):
        width_coefficient, depth_coefficient, _, dropout_rate = (
            efficientnet_params(model_name))
        global_params = efficientnet(
            width_coefficient,
            depth_coefficient,
            dropout_rate,
            normalization_op_params=normalization_op_params)

    elif model_name.startswith('efficientnet-lite'):
        width_coefficient, depth_coefficient, _, dropout_rate = (
            efficientnet_lite_params(model_name))
        global_params = efficientnet_lite(
            width_coefficient,
            depth_coefficient,
            dropout_rate,
            normalization_op_params=normalization_op_params)

    else:
        raise NotImplementedError('model name is not pre-defined: %s' %
                                  model_name)

    if override_params:
        global_params = global_params._replace(**override_params)

    decoder = BlockDecoder()
    blocks_args = decoder.decode(global_params.blocks_args)

    # logging.info('global_params= %s', global_params)
    return blocks_args, global_params


class EfficientNet(tf.keras.Model):
    def __init__(
            self,
            input_shape,
            model_name,
            checkpoint=None,
            override_params=None,
            normalization_op_params=None):

        blocks_args, global_params = get_model_params(
            model_name,
            override_params=override_params,
            normalization_op_params=normalization_op_params)

        input_layer = tf.keras.Input(shape=input_shape, name="efficient_input")
        _model = Model(blocks_args, global_params, model_name)
        features = _model(input_layer, features_only=True)
        outputs = {str(idx): features[idx] for idx in range(2, 6)}

        super(EfficientNet, self).__init__(
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
