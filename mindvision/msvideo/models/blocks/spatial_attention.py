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
""" SpatialAttention for ARN."""

from mindspore import nn
from mindspore import ops

from mindvision.msvideo.models.blocks.unit3d import Unit3D


class SpatialAttention(nn.Cell):
    """
    Initialize spatial attention unit which refine the aggregation step
    by re-weighting block contributions.

    Args:
        in_channels: The number of channels of the input feature.
        out_channels: The number of channels of the output of hidden layers.

    Returns:
        Tensor of shape (1, 1, H, W).
    """

    def __init__(self,
                 in_channels: int = 64,
                 out_channels: int = 16
                 ):
        super(SpatialAttention, self).__init__()
        self.layer1 = Unit3D(in_channels, out_channels)
        self.layer2 = Unit3D(out_channels, out_channels)
        self.max_pool = ops.MaxPool3D(kernel_size=(2, 1, 1), strides=(2, 1, 1))
        self.conv3d = nn.Conv3d(out_channels, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        out = self.layer1(x)
        out = self.max_pool(out)
        out = self.layer2(out)
        out = self.max_pool(out)
        out = self.sigmoid(self.conv3d(out))
        return out
