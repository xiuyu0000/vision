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
"""Block for ConvNeXt."""

import numpy as np

import mindspore as ms
from mindspore import nn, ops
from mindspore import Parameter, Tensor

from mindvision.classification.models.blocks import DropPathWithScale


class ConvNeXtBlock(nn.Cell):
    """
    ConvNext Block. There are two equivalent implementations:
    (1) DwConv -> layernorm(channel_first)->1*1 Conv —>GELU -> 1*1 Conv,all in (N, C, H, W);
    (2) DwConv -> Permute to (NHWC), layernorm(channels_last) -> Dense -> GELU -> Dense,
    permute back to (NCHW). We use (2).

    Args:
        dim(int):Number of input channels.
        drop_prob(float): Stochastic depth rate. Default:0.0.
        layer_scale(float): Init value for Layer Scale. Default:1e-6.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> ConvNeXtBlock(dim=96, drop_prob=0.0, layer_scale=1e-6)
    """

    def __init__(self,
                 dim: int,
                 drop_prob: float = 0.0,
                 layer_scale: float = 1e-6):
        super(ConvNeXtBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, pad_mode="pad", padding=3, group=dim, has_bias=True)
        self.layer_norm = nn.LayerNorm(normalized_shape=(dim,), epsilon=1e-6)
        self.transpose = ops.Transpose()
        self.pwconv1 = nn.Dense(dim, 4 * dim)
        self.acti = nn.GELU()
        self.pwconv2 = nn.Dense(4 * dim, dim)
        if layer_scale > 0.:
            self.gamma = Parameter(Tensor(layer_scale * np.ones((dim,)), dtype=ms.float32), requires_grad=True)
        else:
            self.gamma = Parameter(Tensor(np.ones((dim,)), dtype=ms.float32), requires_grad=False)
        self.drop_path = DropPathWithScale(drop_prob)

    def construct(self, x):
        """ConvNeXtBlock forward construct"""
        shortcut = x
        x = self.dwconv(x)
        x = self.transpose(x, (0, 2, 3, 1))
        x = self.layer_norm(x)
        x = self.pwconv1(x)
        x = self.acti(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = self.transpose(x, (0, 3, 1, 2))
        x = shortcut + self.drop_path(x)
        return x
