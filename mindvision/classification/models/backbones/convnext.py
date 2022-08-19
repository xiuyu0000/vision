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
"""ConvNeXt backbone."""

import mindspore as ms
from mindspore import nn, ops, Tensor

from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.classification.models.blocks import ConvNeXtBlock, DownSample, TransposeChannel

__all__ = ["ConvNeXt"]


@ClassFactory.register(ModuleType.BACKBONE)
class ConvNeXt(nn.Cell):
    """
    Args:
        in_channels(int): Number of input image channels. Default: 3
        depths (List(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (List(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale (float): Init value for Layer Scale. Default: 1e-6.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> backbone = ConvNeXt()
    """

    def __init__(self,
                 in_channels=3,
                 depths=None,
                 dims=None,
                 drop_path_rate=0.,
                 layer_scale=1e-6):
        super(ConvNeXt, self).__init__()
        if not depths:
            depths = [3, 3, 9, 3]
        if not dims:
            dims = [96, 192, 384, 768]
        self.start_cell = nn.SequentialCell([nn.Conv2d(in_channels, dims[0], 4, 4, has_bias=True),
                                             TransposeChannel(target='channel_last'),
                                             nn.LayerNorm(normalized_shape=(dims[0],), epsilon=1e-6),
                                             TransposeChannel(target='channel_first')])
        linspace = ops.LinSpace()
        start = Tensor(0, ms.float32)
        dp_rates = [x.item((0,)) for x in linspace(start, drop_path_rate, sum(depths))]

        self.block1 = nn.SequentialCell([ConvNeXtBlock(dim=dims[0],
                                                       drop_prob=dp_rates[j],
                                                       layer_scale=layer_scale)
                                         for j in range(depths[0])])
        del dp_rates[: depths[0]]

        down_sample_blocks_list = []
        for i in range(3):
            down_sample = DownSample(in_channels=dims[i], out_channels=dims[i+1])
            down_sample_blocks_list.append(down_sample)
            block = nn.SequentialCell([ConvNeXtBlock(dim=dims[i+1],
                                                     drop_prob=dp_rates[j],
                                                     layer_scale=layer_scale)
                                       for j in range(depths[i+1])])
            down_sample_blocks_list.append(block)
            del dp_rates[: depths[i+1]]
        self.down_sample_blocks = nn.SequentialCell(down_sample_blocks_list)

    def construct(self, x):
        x = self.start_cell(x)
        x = self.block1(x)
        x = self.down_sample_blocks(x)
        return x
