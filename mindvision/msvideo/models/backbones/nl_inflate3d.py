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
"""nonlocal backbone."""

from typing import List, Optional, Tuple

from mindspore import ops
from mindspore import nn


from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.msvideo.models.backbones.resnet3d import ResNet3D, ResidualBlockBase3D, ResidualBlock3D
from mindvision.msvideo.models.blocks.unit3d import Unit3D
from mindvision.msvideo.models.blocks.inflate_conv3d import Inflate3D
from mindvision.msvideo.models.blocks.nonlocalblock import NonLocalBlockND

__all__ = [
    'NLInflateBlockBase3D',
    'NLInflateBlock3D',
    'NLInflateResNet3D',
    'NLResInflate3D50',
]



class NLInflateBlockBase3D(ResidualBlockBase3D):
    """
    ResNet residual block base definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        conv12(nn.Cell, optional): Block that constructs first two conv layers. It can be `Inflate3D`,
            `Conv2Plus1D` or other custom blocks, this block should construct a layer where the name
            of output feature channel size is `mid_channel` for the third conv layers. Default: Inflate3D.
        group (int): Group convolutions. Default: 1.
        base_width (int): Width of per group. Default: 64.
        norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
        down_sample (nn.Cell, optional): Downsample structure. Default: None.
        non_local(bool): Determine whether to apply nonlocal block in this block. Default: False.
        non_local_mode(str): Determine which mode to choose for nonlocalblock. Default: dot.
        **kwargs(dict, optional): Key arguments for "conv12", it can contain "stride", "inflate", etc.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> NLInflateBlockBase3D(3, 256)
    """

    expansion: int = 1

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 conv12: Optional[nn.Cell] = Inflate3D,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None,
                 non_local: bool = False,
                 non_local_mode: str = 'dot',
                 **kwargs
                 ) -> None:

        assert group != 1 or base_width == 64, "NLInflateBlockBase3D only supports groups=1 and base_width=64"
        super(NLInflateBlockBase3D, self).__init__(in_channel=in_channel,
                                                   out_channel=out_channel,
                                                   conv12=conv12,
                                                   norm=norm,
                                                   down_sample=down_sample,
                                                   **kwargs)
        self.non_local = non_local
        if self.non_local:
            in_channels = out_channel * self.expansion
            self.non_local_block = NonLocalBlockND(in_channels, mode=non_local_mode)

    def construct(self, x):
        """NLInflateBlockBase3D construct."""
        identity = x

        out = self.conv12(x)

        if self.down_sample:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)
        if self.non_local:
            out = self.non_local_block(out)
        return out


class NLInflateBlock3D(ResidualBlock3D):
    """
    ResNet3D residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        conv12(nn.Cell, optional): Block that constructs first two conv layers. It can be `Inflate3D`,
            `Conv2Plus1D` or other custom blocks, this block should construct a layer where the name
            of output feature channel size is `mid_channel` for the third conv layers. Default: Inflate3D.
        group (int): Group convolutions. Default: 1.
        base_width (int): Width of per group. Default: 64.
        norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
        down_sample (nn.Cell, optional): Downsample structure. Default: None.
        non_local(bool): Determine whether to apply nonlocal block in this block. Default: False.
        non_local_mode(str): Determine which mode to choose for nonlocalblock. Default: dot.
        **kwargs(dict, optional): Key arguments for "conv12", it can contain "stride", "inflate", etc.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> from mindvision.classification.models.backbones import NLInflateBlock3D
        >>> NLInflateBlock3D(3, 256)
    """

    expansion: int = 4

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 conv12: Optional[nn.Cell] = Inflate3D,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None,
                 non_local: bool = False,
                 non_local_mode: str = 'dot',
                 **kwargs
                 ) -> None:
        super(NLInflateBlock3D, self).__init__(in_channel=in_channel,
                                               out_channel=out_channel,
                                               mid_channel=out_channel,
                                               conv12=conv12,
                                               group=group,
                                               norm=norm,
                                               activation=(nn.ReLU, nn.ReLU),
                                               down_sample=down_sample,
                                               **kwargs)
        # conv3d doesn't support group>1 now at 1.6.1 version
        out_channel = int(out_channel * (base_width / 64.0)) * group

        self.non_local = non_local
        if self.non_local:
            in_channels = out_channel * self.expansion
            self.non_local_block = NonLocalBlockND(in_channels, mode=non_local_mode)

    def construct(self, x):
        """NLInflateBlock3D construct."""
        identity = x

        out = self.conv12(x)
        out = self.conv3(out)

        if self.down_sample:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)
        if self.non_local:
            out = self.non_local_block(out)
        return out


class NLInflateResNet3D(ResNet3D):
    """Inflate3D with ResNet3D backbone and non local block.

    Args:
        block (Optional[nn.Cell]): THe block for network.
        layer_nums (list): The numbers of block in different layers.
        stage_channels(tuple): The numbers of channel in different stage.
        stage_strides(tuple): Stride size for ResNet3D convolutional layer.
        down_sample (nn.Cell, optional): Down_sample structure. Default: Unit3D.
        inflate(tuple): Whether to inflate kernel.
        non_local: Determine whether to apply nonlocal block in this block.
        **kwargs(dict, optional): Key arguments for "conv12", it can contain "stride", "inflate", etc.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Returns:
        Tensor, output tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindvision.msvideo.models.backbones.nl_inflate3d import NLInflateResNet3D, NLInflateBlock3D
        >>> net = NLInflateResNet3D(NLInflateBlock3D, [3, 4, 6, 3])
        >>> x = ms.Tensor(np.ones([1, 3, 32, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 2048, 16, 7, 7)
    """

    def __init__(self,
                 block: Optional[nn.Cell],
                 layer_nums: List[int],
                 stage_channels: Tuple[int] = (64, 128, 256, 512),
                 stage_strides: Tuple[Tuple[int]] = ((1, 1, 1),
                                                     (1, 2, 2),
                                                     (1, 2, 2),
                                                     (1, 2, 2)),
                 down_sample: Optional[nn.Cell] = Unit3D,
                 inflate: Tuple[Tuple[int]] = ((1, 1, 1),
                                               (1, 0, 1, 0),
                                               (1, 0, 1, 0, 1, 0),
                                               (0, 1, 0)),
                 non_local: Tuple[Tuple[int]] = ((0, 0, 0),
                                                 (0, 1, 0, 1),
                                                 (0, 1, 0, 1, 0, 1),
                                                 (0, 0, 0)),
                 **kwargs
                 ):
        super(NLInflateResNet3D, self).__init__(block=block,
                                                layer_nums=layer_nums,
                                                stage_channels=stage_channels,
                                                stage_strides=stage_strides,
                                                down_sample=down_sample)
        self.in_channels = stage_channels[0]
        self.conv1 = Unit3D(3, stage_channels[0], kernel_size=(3, 7, 7), stride=(1, 2, 2), norm=self.norm)
        self.maxpool = ops.MaxPool3D(
            kernel_size=(1, 3, 3),
            strides=(1, 2, 2),
            pad_mode="pad",
            pad_list=(0, 0, 1, 1, 1, 1))
        self.pool2 = ops.MaxPool3D(kernel_size=(2, 1, 1), strides=(2, 1, 1))
        self.layer1 = self._make_layer(
            block,
            stage_channels[0],
            layer_nums[0],
            stride=tuple(stage_strides[0]),
            norm=self.norm,
            inflate=inflate[0],
            non_local=non_local[0],
            **kwargs)
        self.layer2 = self._make_layer(
            block,
            stage_channels[1],
            layer_nums[1],
            stride=tuple(stage_strides[1]),
            norm=self.norm,
            inflate=inflate[1],
            non_local=non_local[1],
            **kwargs)
        self.layer3 = self._make_layer(
            block,
            stage_channels[2],
            layer_nums[2],
            stride=tuple(stage_strides[2]),
            norm=self.norm,
            inflate=inflate[2],
            non_local=non_local[2],
            **kwargs)
        self.layer4 = self._make_layer(
            block,
            stage_channels[3],
            layer_nums[3],
            stride=tuple(stage_strides[3]),
            norm=self.norm,
            inflate=inflate[3],
            non_local=non_local[3],
            **kwargs)

    def construct(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.pool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


@ClassFactory.register(ModuleType.BACKBONE)
class NLResInflate3D50(NLInflateResNet3D):
    """
    The class of ResNet50 uses the registration mechanism to register, need to use the yaml configuration file to call.
    """

    def __init__(self, **kwargs):
        super(NLResInflate3D50, self).__init__(NLInflateBlock3D, [3, 4, 6, 3], **kwargs)
