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
""" TemporalAttention for ARN."""

from mindspore import nn

from mindvision.msvideo.engine.ops import MaxPool3D
from mindvision.msvideo.models.blocks.unit3d import Unit3D


class TemporalAttention(nn.Cell):
    """
    Initialize temporal attention unit which refine the aggregation step
    by re-weighting block contributions.

    Args:
        in_channels: The number of channels of the input feature. Default: 64.
        out_channels: The number of channels of the output of hidden layers. Default: 16.

    Returns:
        Tensor of shape (1, T, 1, 1).
    """

    def __init__(self,
                 in_channels: int = 64,
                 out_channels: int = 16
                 ):
        super(TemporalAttention, self).__init__()
        self.ta1 = nn.SequentialCell(
            Unit3D(in_channels=in_channels, out_channels=out_channels),
            MaxPool3D(kernel_size=(1, 4, 4), strides=(1, 4, 4)),
            Unit3D(in_channels=out_channels, out_channels=out_channels),
            MaxPool3D(kernel_size=(1, 8, 8), strides=(1, 8, 8)),
            nn.Conv3d(out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid())

    def construct(self, x):
        ta = self.ta1(x)
        return ta
