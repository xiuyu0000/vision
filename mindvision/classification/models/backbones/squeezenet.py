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
"""SqueezeNet1_0 backbone."""
from mindspore import nn
from mindspore.common import initializer as weight_init
from mindspore.ops import operations as P
from mindvision.classification.models.blocks import ConvNormActivation
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = [
    "Fire",
    "SqueezeNet",
    "SqueezeNetV0",
    "SqueezeNetV1"
]


class Fire(nn.Cell):
    """
    SqueezeNet Fire network definition.

    Args:
        inplanes (int): The input channel.
        squeeze_planes (int): The output channel.
        expand1x1_planes (int): 1x1 convolutional layer.
        expand3x3_planes (int): 3x3 convolutional layer.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> from mindvision.classification.models.backbones import Fire
        >>> Fire(96, 16, 64, 64),
    """

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.squeeze = ConvNormActivation(
            inplanes,
            squeeze_planes,
            kernel_size=1,
            has_bias=True)
        self.expand1x1 = ConvNormActivation(
            squeeze_planes,
            expand1x1_planes,
            kernel_size=1,
            has_bias=True)
        """
        To make the output of the 1x1 and 3x3 filters the same size,
        add a pixel boundary to the original input of the 3x3 filter in the expand modules.
        (zero-padding)
        """
        self.expand3x3 = ConvNormActivation(
            squeeze_planes,
            expand3x3_planes,
            kernel_size=3,
            has_bias=True)
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        """Fire module construct"""
        x = self.squeeze(x)
        return self.concat((self.expand1x1(x), self.expand3x3(x)))


class SqueezeNet(nn.Cell):
    """
    SqueezeNet architecture.

    Args:
        version (str): The version of SqueezeNet, '1_0' or '1_1'. Default: 1_0.
        num_classes (int): Number of categories of classification. Default: 10.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, output tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> net = SqueezeNet('1_0',1000)

    About SqueezeNet:

    SqueezeNet is a lightweight and efficient CNN model proposed by F.N. Iandola et al.
    that can compress the original Alexnet to 1/510th of its original size (<0.5MB) without any loss of accuracy.
    The core of SqueezeNet is the Fire module, which consists of squeeze and expand.
    squeeze uses a 1X1 convolution kernel to convolve the upper featuremap layer, reducing the featuremap dimension.

    Citation:

    .. code-block::

        @article{SqueezeNet,
        Author = {Forrest N. Iandola and Song Han and Matthew W. Moskewicz and Khalid Ashraf and
                  William J. Dally and Kurt Keutzer},
        Title = {SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and < 0.5MB model size},
        Journal = {arXiv:1602.07360},
        Year = {2016}
}
    """

    def __init__(self, version='1_0', num_classes=10):
        super(SqueezeNet, self).__init__()
        if version == '1_0':
            self.features = nn.SequentialCell([
                nn.Conv2d(3,
                          96,
                          kernel_size=7,
                          stride=2,
                          pad_mode='valid',
                          has_bias=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(512, 64, 256, 256),
            ])
        elif version == '1_1':
            self.features = nn.SequentialCell([
                nn.Conv2d(3,
                          64,
                          kernel_size=3,
                          stride=2,
                          pad_mode='pad',
                          padding=1,
                          has_bias=True),  # In PyTorch version, padding=1
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                # inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            ])
        else:
            raise ValueError(f"Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        self.final_conv = nn.Conv2d(512,
                                    num_classes,
                                    kernel_size=1,
                                    has_bias=True)
        self.dropout = nn.Dropout(keep_prob=0.5)
        self.relu = nn.ReLU()
        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.custom_init_weight()

    def custom_init_weight(self):
        """
        Init the weight of Conv2d in the net.
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                if cell is self.final_conv:
                    cell.weight.set_data(
                        weight_init.initializer('normal', cell.weight.shape,
                                                cell.weight.dtype))
                else:
                    cell.weight.set_data(
                        weight_init.initializer('he_uniform',
                                                cell.weight.shape,
                                                cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        weight_init.initializer('zeros', cell.bias.shape,
                                                cell.bias.dtype))

    def construct(self, x):
        """Fire Module construct."""
        x = self.features(x)
        x = self.dropout(x)
        x = self.final_conv(x)
        x = self.relu(x)
        x = self.mean(x, (2, 3))
        x = self.flatten(x)
        return x


@ClassFactory.register(ModuleType.BACKBONE)
class SqueezeNetV0(SqueezeNet):
    """
    The class of SqueezeNetV0 uses the registration mechanism to register,
    need to use the yaml configuration file to call.
    """

    def __init__(self, **kwargs):
        super(SqueezeNetV0, self).__init__(
            version='1_0', num_classes=1000, **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class SqueezeNetV1(SqueezeNet):
    """
    The class of SqueezeNetV1 uses the registration mechanism to register,
    need to use the yaml configuration file to call.
    """

    def __init__(self, **kwargs):
        super(SqueezeNetV1, self).__init__(
            version='1_1', num_classes=1000, **kwargs)
