# Copyright 2020 Huawei Technologies Co., Ltd
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
"""MobileNetV3 backbone."""

from typing import Optional, List

from mindspore import nn, ops

from mindvision.classification.models.utils import make_divisible
from mindvision.classification.models.blocks import ConvNormActivation, SqueezeExcite
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = [
    "MobileNetV3",
]


class ResUnit(nn.Cell):
    """
    ResUnit warpper definition.

    Args:
        num_in (int): Input channel.
        num_mid (int): Middle channel.
        num_out (int): Output channel.
        kernel_size (int): kernel size for the second convolutional layer.
        norm: Norm layer that will be stacked on top of the convoution layer.
        activation (str): Activation function which will be stacked on top of the
        normalization layer (if not None), otherwise on top of the conv layer.
        stride (int): Stride size for the second convolutional layer. Default: 1.
        use_se (bool): Use SE warpper or not. Default: False.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResUnit(3, 16, 16, kernel_size=3, norm=nn.BatchNorm2d, activation='relu')
    """

    def __init__(self, num_in: int,
                 num_mid: int,
                 num_out: int,
                 kernel_size: int,
                 norm: nn.Cell,
                 activation: str,
                 stride: int = 1,
                 use_se: bool = False) -> None:
        super(ResUnit, self).__init__()
        self.use_se = use_se
        self.use_short_cut_conv = num_in == num_out and stride == 1
        self.use_hs = activation == 'hswish'
        self.activation = nn.HSwish if self.use_hs else nn.ReLU

        layers = []

        # Expand.
        if num_in != num_mid:
            layers.append(
                ConvNormActivation(num_in, num_mid, kernel_size=1, norm=norm, activation=self.activation)
            )

        # DepthWise.
        layers.append(
            ConvNormActivation(num_mid, num_mid, kernel_size=kernel_size, stride=stride, groups=num_mid, norm=norm,
                               activation=self.activation)
        )
        if use_se:
            squeeze_channel = make_divisible(num_mid // 4, 8)
            layers.append(
                SqueezeExcite(num_mid, squeeze_channel, nn.ReLU, nn.HSigmoid)
            )

        # Project.
        layers.append(
            ConvNormActivation(num_mid, num_out, kernel_size=1, norm=norm, activation=None)
        )

        self.block = nn.SequentialCell(layers)
        self.add = ops.Add()

    def construct(self, x):
        out = self.block(x)

        if self.use_short_cut_conv:
            out = self.add(out, x)

        return out


@ClassFactory.register(ModuleType.BACKBONE)
class MobileNetV3(nn.Cell):
    """
    MobileNetV3 architecture.

    Args:
        model_cfgs (List): config of models, large or small.
        multiplier (float): Channels multiplier for round to 8/16 and others. Default is 1.0. Default: 1.0.
        norm (nn.Cell, optional): Norm layer that will be stacked on top of the convoution
        layer. Default: None.
        round_nearest (int): Round the number of channels in each layer to be a multiple of this number
        set to 1 to turn off rounding. Default is 8.

    Returns:
        Tensor, output tensor.
    """

    def __init__(self,
                 model_cfgs: List,
                 multiplier: float = 1.0,
                 norm: Optional[nn.Cell] = None,
                 round_nearest: int = 8,
                 ) -> None:
        super(MobileNetV3, self).__init__()

        if not norm:
            norm = nn.BatchNorm2d

        self.inplanes = 16
        layers = []

        # Building first layer.
        first_conv_in_channel = 3
        first_conv_out_channel = make_divisible(self.inplanes * multiplier, round_nearest)
        layers.append(
            ConvNormActivation(
                first_conv_in_channel,
                first_conv_out_channel,
                kernel_size=3,
                stride=2,
                norm=norm,
                activation=nn.HSwish
            )
        )

        # Building inverted residual blocks.
        for layer_cfg in model_cfgs:
            layers.append(self._make_layer(kernel_size=layer_cfg[0],
                                           exp_ch=make_divisible(multiplier * layer_cfg[1], round_nearest),
                                           out_channel=make_divisible(multiplier * layer_cfg[2], round_nearest),
                                           use_se=layer_cfg[3],
                                           activation=layer_cfg[4],
                                           stride=layer_cfg[5],
                                           norm=norm
                                           )
                          )

        lastconv_input_channel = make_divisible(multiplier * model_cfgs[-1][2], round_nearest)
        lastconv_output_channel = lastconv_input_channel * 6

        # Building last several layers.
        layers.append(
            ConvNormActivation(
                lastconv_input_channel,
                lastconv_output_channel,
                kernel_size=1,
                norm=norm,
                activation=nn.HSwish
            )
        )

        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        x = self.features(x)

        return x

    def _make_layer(self,
                    kernel_size: int,
                    exp_ch: int,
                    out_channel: int,
                    use_se: bool,
                    activation: str,
                    norm: nn.Cell,
                    stride: int = 1
                    ):
        """Block layers."""
        layer = ResUnit(self.inplanes, exp_ch, out_channel,
                        kernel_size=kernel_size, stride=stride, activation=activation, use_se=use_se, norm=norm)
        self.inplanes = out_channel

        return layer
