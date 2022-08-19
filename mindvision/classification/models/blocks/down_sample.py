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
""" DownSample module"""
from mindspore import nn, ops


class DownSample(nn.Cell):
    """
    Down sample block for ConvNeXt, composed with layer norm and conv2d.

    Args:
        in_channels(int): Number of input channels.
        out_channels(int): Number of output channels.
        kernel_size(int): Convolution kernel size. Default: 2.
        stride(int): stride size. Default: 2.
        eps(float): A value added to the denominator for numerical stability. Default: 1e-6.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> DownSample(in_channels=96, out_channels=96, kernel_size=2, stride=2, eps=1e-6)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: int = 2,
                 eps: float = 1e-6):
        super(DownSample, self).__init__()
        self.transpose = ops.Transpose()
        self.layer_norm = nn.LayerNorm(normalized_shape=(in_channels,), epsilon=eps)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, has_bias=True)

    def construct(self, x):
        """DownSample forward construct"""
        x = self.transpose(x, (0, 2, 3, 1))
        x = self.layer_norm(x)
        x = self.transpose(x, (0, 3, 1, 2))
        x = self.conv(x)
        return x
