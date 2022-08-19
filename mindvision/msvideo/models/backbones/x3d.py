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
"""x3d backbone."""

import math
from typing import List, Optional, Tuple

from mindspore import nn

from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.msvideo.models.backbones.resnet3d import ResNet3D, ResidualBlock3D
from mindvision.msvideo.models.blocks.unit3d import Unit3D
from mindvision.msvideo.models.blocks.inflate_conv3d import Inflate3D
from mindvision.classification.engine.ops.swish import Swish
from mindvision.msvideo.models.blocks.squeeze_excite3d import SqueezeExcite3D
from mindvision.msvideo.utils.others import drop_path
import mindvision.msvideo.utils.init_weight as init_weights

__all__ = [
    'BlockX3D',
    'ResNetX3D',
    'X3DM',
    'X3DS',
    'X3DXS',
    'X3DL',
]


class BlockX3D(ResidualBlock3D):
    """
    BlockX3D 3d building block for X3D.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        conv12(nn.Cell, optional): Block that constructs first two conv layers.
            It can be `Inflate3D`, `Conv2Plus1D` or other custom blocks, this
            block should construct a layer where the name of output feature channel
            size is `mid_channel` for the third conv layers. Default: Inflate3D.
        inflate (bool): Whether to inflate kernel.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        down_sample (nn.Module | None): DownSample layer. Default: None.
        block_idx (int): the id of the block.
        se_ratio (float | None): The reduction ratio of squeeze and excitation
            unit. If set as None, it means not using SE unit. Default: None.
        use_swish (bool): Whether to use swish as the activation function
            before and after the 3x3x3 conv. Default: True.
        drop_connect_rate (float): dropout rate. If equal to 0.0, perform no dropout.
        bottleneck_factor (float): Bottleneck expansion factor for the 3x3x3 conv.
    """
    expansion: int = 1

    def __init__(self,
                 in_channel,
                 out_channel,
                 conv12: Optional[nn.Cell] = Inflate3D,
                 inflate: bool = False,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None,
                 block_idx: int = 0,
                 se_ratio: float = 0.0625,
                 use_swish: bool = True,
                 drop_connect_rate: float = 0.0,
                 bottleneck_factor: float = 2.25,
                 **kwargs):

        super(BlockX3D, self).__init__(in_channel=in_channel,
                                       out_channel=out_channel,
                                       mid_channel=int(out_channel * bottleneck_factor),
                                       conv12=conv12,
                                       norm=norm,
                                       down_sample=down_sample,
                                       inflate=inflate,
                                       **kwargs)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.se_ratio = se_ratio
        self.use_swish = use_swish
        self._drop_connect_rate = drop_connect_rate
        if self.use_swish:
            self.swish = Swish()

        self.se_module = None
        if self.se_ratio > 0.0 and (block_idx + 1) % 2:
            self.se_module = SqueezeExcite3D(self.conv12.mid_channel, self.se_ratio)

        self.conv3 = Unit3D(
            in_channels=self.conv12.mid_channel,
            out_channels=self.out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            norm=nn.BatchNorm3d,
            activation=None)
        self.conv3.transform_final_bn = True

    def construct(self, x):
        """Defines the computation performed at every call."""
        identity = x

        out = self.conv12(x)
        if self.se_module is not None:
            out = self.se_module(out)
        if self.use_swish:
            out = self.swish(out)

        out = self.conv3(out)

        if self.training and self._drop_connect_rate > 0.0:
            out = drop_path(out, self._drop_connect_rate)

        if self.down_sample:
            identity = self.down_sample(x)

        out = out + identity
        out = self.relu(out)

        return out


@ClassFactory.register(ModuleType.BACKBONE)
class ResNetX3D(ResNet3D):
    """
    X3D backbone definition.

    Args:
        block (Optional[nn.Cell]): THe block for network.
        layer_nums (list): The numbers of block in different layers.
        stage_channels (Tuple[int]): Output channel for every res stage.
        stage_strides (Tuple[Tuple[int]]): Stride size for ResNet3D convolutional layer.
        drop_rates (list): list of the drop rate in different blocks. The basic rate at which blocks
            are dropped, linearly increases from input to output blocks.
        down_sample (Optional[nn.Cell]): Residual block in every resblock, it can transfer the input
            feature into the same channel of output. Default: Unit3D.
        bottleneck_factor (float): Bottleneck expansion factor for the 3x3x3 conv.
        fc_init_std (float): The std to initialize the fc layer(s).

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Returns:
        Tensor, output tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> net = ResNetX3D(BlockX3D, [3, 5, 11, 7], (24, 48, 96, 192), ((1, 2, 2),(1, 2, 2),
        >>>             (1, 2, 2),(1, 2, 2)), [0.2, 0.3, 0.4, 0.5], Unit3D)
    """

    def __init__(self,
                 block: Optional[nn.Cell],
                 layer_nums: List[int],
                 stage_channels: Tuple[int],
                 stage_strides: Tuple[Tuple[int]],
                 drop_rates: List[float],
                 down_sample: Optional[nn.Cell] = Unit3D,
                 bottleneck_factor: float = 2.25,
                 fc_init_std: float = 0.01
                 ):
        super(ResNetX3D, self).__init__(block=block,
                                        layer_nums=layer_nums,
                                        stage_channels=stage_channels,
                                        stage_strides=stage_strides,
                                        down_sample=down_sample)
        self.in_channels = stage_channels[0]
        self.base_channels = 24
        self.conv1 = nn.SequentialCell([Unit3D(3,
                                               self.base_channels,
                                               kernel_size=(1, 3, 3),
                                               stride=(1, 2, 2),
                                               norm=None,
                                               activation=None),
                                        Unit3D(self.base_channels,
                                               self.base_channels,
                                               kernel_size=(5, 1, 1),
                                               stride=(1, 1, 1))])
        self.layer1 = self._make_layer(
            block,
            stage_channels[0],
            layer_nums[0],
            stride=tuple(stage_strides[0]),
            inflate=False,
            drop_connect_rate=drop_rates[0],
            block_idx=list(range(layer_nums[0])))
        self.layer2 = self._make_layer(
            block,
            stage_channels[1],
            layer_nums[1],
            stride=tuple(stage_strides[1]),
            inflate=False,
            drop_connect_rate=drop_rates[1],
            block_idx=list(range(layer_nums[1])))
        self.layer3 = self._make_layer(
            block,
            stage_channels[2],
            layer_nums[2],
            stride=tuple(stage_strides[2]),
            inflate=False,
            drop_connect_rate=drop_rates[2],
            block_idx=list(range(layer_nums[2])))
        self.layer4 = self._make_layer(
            block,
            stage_channels[3],
            layer_nums[3],
            stride=tuple(stage_strides[3]),
            inflate=False,
            drop_connect_rate=drop_rates[3],
            block_idx=list(range(layer_nums[3])))
        self.conv5 = Unit3D(stage_channels[-1],
                            int(math.ceil(stage_channels[-1] * bottleneck_factor)),
                            kernel_size=1,
                            stride=1,
                            padding=0)
        init_weights.init_weights(self, fc_init_std, True)

    def construct(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        return x


@ClassFactory.register(ModuleType.BACKBONE)
class X3DM(ResNetX3D):
    """
    The class of X3D_M uses the registration mechanism to register, need to use
    the yaml configuration file to call.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(self, layer_nums, **kwargs):
        super(X3DM, self).__init__(BlockX3D, layer_nums, (24, 48, 96, 192),
                                   ((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
                                   [0.2, 0.3, 0.4, 0.5], **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class X3DS(ResNetX3D):
    """
    The class of X3D_S uses the registration mechanism to register, need to use
    the yaml configuration file to call.
    """

    def __init__(self, layer_nums, **kwargs):
        super(X3DS, self).__init__(BlockX3D, layer_nums, (24, 48, 96, 192),
                                   ((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
                                   [0.2, 0.3, 0.4, 0.5], **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class X3DXS(ResNetX3D):
    """
    The class of X3D_XS uses the registration mechanism to register, need to use
    the yaml configuration file to call.
    """

    def __init__(self, layer_nums, **kwargs):
        super(X3DXS, self).__init__(BlockX3D, layer_nums, (24, 48, 96, 192),
                                    ((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
                                    [0.2, 0.3, 0.4, 0.5], **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class X3DL(ResNetX3D):
    """
    The class of X3D_L uses the registration mechanism to register, need to use
    the yaml configuration file to call.
    """

    def __init__(self, layer_nums, **kwargs):
        super(X3DL, self).__init__(BlockX3D, layer_nums, (24, 48, 96, 192),
                                   ((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
                                   [0.2, 0.3, 0.4, 0.5], **kwargs)
