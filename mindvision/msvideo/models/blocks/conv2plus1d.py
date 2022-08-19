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
"""Conv2Plus1D block."""

from mindspore import nn

from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.msvideo.models.blocks.unit3d import Unit3D


@ClassFactory.register(ModuleType.BACKBONE)
class Conv2Plus1D(nn.Cell):
    """R(2+1)d conv12 block. It implements spatial-temporal feature extraction in
        a sperated way.

    Args:
        in_channels (int):  The number of channels of input frame images.
        out_channels (int):  The number of channels of output frame images.
        kernel_size (tuple): The size of the spatial-temporal convolutional layer kernels.
        stride (Union[int, Tuple[int]]): Stride size for the convolutional layer. Default: 1.
        group (int): Splits filter into groups, in_channels and out_channels must be divisible by the number
            of groups. Default: 1.
        norm (Optional[nn.Cell]): Norm layer that will be stacked on top of the convolution
            layer. Default: nn.BatchNorm3d.
        activation (Optional[nn.Cell]): Activation function which will be stacked on top of the
            normalization layer (if not None), otherwise on top of the conv layer. Default: nn.ReLU.

    Returns:
        Tensor, its channel size is calculated from in_channel, out_channel and kernel_size.
    """

    def __init__(self,
                 in_channel,
                 mid_channel,
                 out_channel,
                 kernel_size=(3, 3, 3),
                 stride=(1, 1, 1),
                 norm=nn.BatchNorm3d,
                 activation=nn.ReLU,
                 ):
        super(Conv2Plus1D, self).__init__()
        self.mid_channel = mid_channel
        if not self.mid_channel:
            self.mid_channel = (in_channel * out_channel * kernel_size[1] * kernel_size[2] * 3) // \
                (in_channel * kernel_size[1] * kernel_size[2] + 3 * out_channel)
        self.conv1 = Unit3D(in_channel,
                            self.mid_channel,
                            kernel_size=(1, kernel_size[1], kernel_size[2]),
                            stride=(1, stride[1], stride[2]),
                            norm=norm,
                            activation=activation)
        self.conv2 = Unit3D(self.mid_channel,
                            out_channel,
                            kernel_size=(kernel_size[0], 1, 1),
                            stride=(stride[0], 1, 1),
                            norm=norm,
                            activation=None)

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
