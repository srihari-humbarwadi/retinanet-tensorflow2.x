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

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.tpu import tpu_function


class TpuBatchNormalization(tf.keras.layers.BatchNormalization):
    """Cross replica batch normalization."""

    def __init__(self, fused=False, **kwargs):
        if fused in (True, None):
            raise ValueError('TpuBatchNormalization does not support fused=True.')
        super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

    def _cross_replica_average(self, t, num_shards_per_group):
        """Calculates the average value of input tensor across TPU replicas."""
        num_shards = tpu_function.get_tpu_context().number_of_shards
        group_assignment = None
        if num_shards_per_group > 1:
            if num_shards % num_shards_per_group != 0:
                raise ValueError(
                    'num_shards: %d mod shards_per_group: %d, should be 0' %
                    (num_shards, num_shards_per_group))
            num_groups = num_shards // num_shards_per_group
            group_assignment = [[
                x for x in range(num_shards) if x // num_shards_per_group == y
            ] for y in range(num_groups)]
        return tf.compat.v1.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
            num_shards_per_group, t.dtype)

    def _moments(self, inputs, reduction_axes, keep_dims):
        """Compute the mean and variance: it overrides the original _moments."""
        shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
            inputs, reduction_axes, keep_dims=keep_dims)

        num_shards = tpu_function.get_tpu_context().number_of_shards or 1
        num_shards_per_group = max(8, num_shards // 8)

        if num_shards_per_group > 1:
            # Compute variance using: Var[X]= E[X^2] - E[X]^2.
            shard_square_of_mean = tf.math.square(shard_mean)
            shard_mean_of_square = shard_variance + shard_square_of_mean
            group_mean = self._cross_replica_average(
                shard_mean, num_shards_per_group)
            group_mean_of_square = self._cross_replica_average(
                shard_mean_of_square, num_shards_per_group)
            group_variance = group_mean_of_square - tf.math.square(group_mean)
            return (group_mean, group_variance)
        else:
            return (shard_mean, shard_variance)

    # Hack to allow loading batch norm weights into custom layer
    def _init_set_name(self, name, zero_based=True):
        if not name:
            self._name = backend.unique_object_name(
                name='batch_normalization',
                zero_based=zero_based)
        else:
            backend.observe_object_name(name)
            self._name = name
