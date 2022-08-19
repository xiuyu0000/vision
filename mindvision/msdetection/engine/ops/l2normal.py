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
"""L2Norm"""

import mindspore as ms
from mindspore import nn
from mindspore import Parameter, ops

__all__ = ["L2Norm"]


class L2Norm(nn.Cell):
    """
    L2Normalization architecture implementation.

    Args:
        n_channels (int): The number of input channel.
        scale (float): The number of scale. Default: 1.0

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Examples:
        >>> L2Norm(n_channels=256, scale=10)
    """

    def __init__(self, n_channels, scale=1.0):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = Parameter(ops.Zeros()((self.n_channels), ms.float32) + self.scale)

    def construct(self, x):
        """L2Norm construct"""
        a = ops.Pow()(x, 2)
        b = a.sum(axis=1, keepdims=True)
        norm = ops.Sqrt()(b) + self.eps
        x = x / norm * self.weight.reshape([1, -1, 1, 1])
        return x
