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
"""Pooling and layer norm neck."""

from mindspore import nn
from mindspore import ops

from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.NECK)
class AvgPoolingLayerNorm(nn.Cell):
    """
    Global avg pooling and layer norm definition.

    Args:
        num_channels (int): The channels num of layer normalization.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> AvgPoolingLayerNorm(num_channels=256)
    """

    def __init__(self,
                 num_channels,
                 ) -> None:
        super(AvgPoolingLayerNorm, self).__init__()
        self.mean = ops.ReduceMean()
        self.layer_norm = nn.LayerNorm(normalized_shape=(num_channels,))

    def construct(self, x):
        x = self.mean(x, (2, 3))
        x = self.layer_norm(x)
        return x
