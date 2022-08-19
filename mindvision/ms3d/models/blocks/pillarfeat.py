# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""PillarsFeat"""

from mindspore import nn
from mindspore import numpy as mnp
from mindspore import ops
from mindspore.common import dtype as mstype


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor"""

    actual_num = ops.ExpandDims()(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = mnp.arange(0, max_num, dtype=mstype.int32).view(*max_num_shape)
    paddings_indicator = actual_num > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class PFNLayer(nn.Cell):
    """PFN layer"""

    def __init__(self, in_channels, out_channels, use_norm, last_layer):
        super(PFNLayer, self).__init__()

        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2

        self.units = out_channels
        self.use_norm = use_norm

        if use_norm:
            self.norm = nn.BatchNorm2d(self.units, eps=1e-3, momentum=0.99)
        else:
            self.norm = ops.Identity()
        self.linear = nn.Dense(in_channels, self.units, has_bias=not use_norm)

        self.transpose = ops.Transpose()
        self.tile = ops.Tile()
        self.concat = ops.Concat(axis=2)
        self.expand_dims = ops.ExpandDims()
        self.argmax_w_value = ops.ArgMaxWithValue(axis=-1, keep_dims=True)

    def construct(self, inputs):
        """forward graph"""
        x = self.linear(inputs)
        x = self.norm(x.transpose((0, 3, 1, 2)))  # [bs, V, P, 4]
        x = ops.ReLU()(x)
        x_max = self.argmax_w_value(x)[1].transpose(
            (0, 2, 3, 1))  # [bs, V, P, 4]
        if self.last_vfe:
            return x_max
        x_repeat = self.tile(
            x_max, (1, 1, inputs.shape[1], 1))  # [bs, V, P, 4]
        x_concatenated = self.concat([x, x_repeat])
        return x_concatenated


class PillarFeatureNet(nn.Cell):
    """Pillar feature net"""

    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        super(PillarFeatureNet, self).__init__()
        num_input_features += 5

        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []

        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True

            pfn_layers.append(
                PFNLayer(in_filters, out_filters,
                         use_norm, last_layer=last_layer)
            )
        self.pfn_layers = nn.SequentialCell(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self.expand_dims = ops.ExpandDims()

    def construct(self, features, num_points, coors):
        """forward graph"""
        bs, v, _, _ = features.shape
        points_mean = (features[:, :, :, :3].sum(axis=2, keepdims=True) /
                       ops.Maximum()(num_points, 1).view(bs, v, 1, 1))
        f_cluster = features[:, :, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = ops.ZerosLike()(features[:, :, :, :2])
        f_center[:, :, :, 0] = features[:, :, :, 0] - (
            self.expand_dims(coors[:, :, 2].astype(mstype.float32), 2) * self.vx + self.x_offset)
        f_center[:, :, :, 1] = features[:, :, :, 1] - (
            self.expand_dims(coors[:, :, 1].astype(mstype.float32), 2) * self.vy + self.y_offset)

        # Combine feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = mnp.norm(features[:, :, :, :3], 2, 3, keepdims=True)
            features_ls.append(points_dist)
        features = ops.Concat(axis=-1)(features_ls)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zero.
        voxel_count = features.shape[2]
        mask = get_paddings_indicator(num_points, voxel_count, axis=1)
        mask = self.expand_dims(mask, -1).astype(features.dtype)
        features *= mask
        # Forward pass through PFNLayers
        features = self.pfn_layers(features)
        return features.squeeze(axis=2)
