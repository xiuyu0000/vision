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
""" DenseNet Backbone """
from collections import OrderedDict
from mindspore import ops
from mindspore import nn
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = [
    'DenseNet',
    'DenseNet121',
    'DenseNet169',
    'DenseNet201',
    'DenseNet232',
    'DenseNet264'
]


def conv7x7(in_channels, out_channels, stride=1, padding=3, has_bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


def conv3x3(in_channels, out_channels, stride=1, padding=1, has_bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


def conv1x1(in_channels, out_channels, stride=1, padding=0, has_bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


class DenseLayer(nn.Cell):
    """
    Dense layer in DenseNet

    Args:
        num_input_features (int): Input Channel
        growth_rate (int): growth rate between Conv Layers
        bn_size (int): bottleneck size
        drop_rate (float): dropout rate

    Returns:
        Tensor, output tensor.

    Examples:
        >>> from mindvision.classification.models.backbones import DenseLayer
        >>> DenseLayer(32, 256, 32, 4, 0.0)
    """

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU()
        self.conv1 = conv1x1(num_input_features, bn_size * growth_rate)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(bn_size * growth_rate, growth_rate)

        self.dropout = nn.Dropout(keep_prob=1.0 - drop_rate)

    def construct(self, features):
        bottleneck = self.conv1(self.relu1(self.norm1(features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck)))
        return self.dropout(new_features)


class DenseBlock(nn.Cell):
    """
    Dense block in DenseNet

    Args:
        num_layers (int): number of dense layers in a block
        num_input_features (int): input features size
        bn_size (int): bottleneck size
        growth_rate (int): growth rate between Conv Layers
        drop_rate (float): dropout rate

    Returns:
        Tensor, output tensor.

    Examples:
        >>> from mindvision.classification.models.backbones import DenseBlock
        >>> DenseLayer(6, 256, 4, 32, 0.0)
    """

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        self.cell_list = nn.CellList()
        for i in range(num_layers):
            self.cell_list.append(
                DenseLayer(
                    num_input_features=num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate
                )
            )
        self.concate = ops.Concat(axis=1)

    def construct(self, features):
        for layer in self.cell_list:
            features = self.concate((features, layer(features)))
        return features


class Transition(nn.Cell):
    """
    Transition layer in dense net

    Args:
        num_input_features (int): input features size
        num_output_features (int): output features size
        avgpool (bool): apply average pooling or max pooling

    Returns:
        Tensor, output tensor.

    Examples:
        >>> from mindvision.classification.models.backbones import Transition
        >>> Transition(224, 256, avgpool=False)
    """

    def __init__(self, num_input_features, num_output_features, avgpool=False):
        super(Transition, self).__init__()
        if avgpool:
            pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features = nn.SequentialCell(OrderedDict([
            ('norm', nn.BatchNorm2d(num_input_features)),
            ('relu', nn.ReLU()),
            ('conv', conv1x1(num_input_features, num_output_features)),
            ('pool', pool)
        ]))

    def construct(self, features):
        return self.features(features)


class DenseNet(nn.Cell):
    """
    DenseNet architecture.

    Args:
        growth_rate (int): growth rate between layers
        block_config  (List[int]): number of layers in blocks
        num_init_features (int|None): whether to apply Transition layer and initial feature size
        bn_size (int): size of bottleneck
        drop_rate (float): dropout rate

    Outputs:
        Tensor, output tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindvision.classification.models.backbones import DenseNet
        >>> net = DenseNet(32, [6, 12, 24, 16], 64)

    Citation:

    .. code-block::


        @inproceedings{huang2017densely,
        title={Densely Connected Convolutional Networks},
        author={Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q },
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2017}
        }
    """
    __constants__ = ['features']

    def __init__(self, growth_rate, block_config, num_init_features=None, bn_size=4, drop_rate=float(0)):
        super(DenseNet, self).__init__()

        layers = OrderedDict()
        if num_init_features:
            num_features = num_init_features
            layers['conv0'] = conv7x7(3, num_features, stride=2, padding=3)
            layers['norm0'] = nn.BatchNorm2d(num_features)
            layers['relu0'] = nn.ReLU()
            layers['pool0'] = nn.MaxPool2d(
                kernel_size=3, stride=2, pad_mode='same')
        else:
            num_features = growth_rate * 2
            layers['conv0'] = conv3x3(3, num_features, stride=1, padding=1)
            layers['norm0'] = nn.BatchNorm2d(num_features)
            layers['relu0'] = nn.ReLU()

        for i, num_layers in enumerate(block_config):
            layers[f"denseblock{i + 1}"] = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            num_features = num_features + num_layers * growth_rate

            if i == len(block_config) - 1:
                continue
            if num_init_features:
                layers[f"transition{i + 1}"] = Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    avgpool=False
                )
                num_features = num_features // 2

        layers['norm5'] = nn.BatchNorm2d(num_features)
        layers['relu5'] = nn.ReLU()

        self.features = nn.SequentialCell(layers)
        self.out_channels = num_features

    def construct(self, inputs):
        return self.features(inputs)

    def get_out_channels(self):
        return self.out_channels


@ClassFactory.register(ModuleType.BACKBONE)
class DenseNet121(DenseNet):
    def __init__(self, **kwargs):
        super(DenseNet121, self).__init__(32, [6, 12, 24, 16], 64, **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class DenseNet161(DenseNet):
    def __init__(self, **kwargs):
        super(DenseNet161, self).__init__(48, [6, 12, 36, 24], 96, **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class DenseNet169(DenseNet):
    def __init__(self, **kwargs):
        super(DenseNet169, self).__init__(32, [6, 12, 32, 32], 64, **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class DenseNet201(DenseNet):
    def __init__(self, **kwargs):
        super(DenseNet201, self).__init__(32, [6, 12, 48, 32], 64, **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class DenseNet232(DenseNet):
    def __init__(self, **kwargs):
        super(DenseNet232, self).__init__(48, [6, 32, 48, 48], 96, **kwargs)


@ClassFactory.register(ModuleType.BACKBONE)
class DenseNet264(DenseNet):
    def __init__(self, **kwargs):
        super(DenseNet264, self).__init__(32, [6, 12, 64, 48], 64, **kwargs)
