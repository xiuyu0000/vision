# Copyright 2022
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
"""squeezenet."""
from mindvision.classification.models.backbones import SqueezeNet
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel

__all__ = [
    "squeezenet_v0",
    "squeezenet_v1"
]


def _squeezenet(arch: str, version: str,
                num_classes: int = 1000, pretrained: bool = False) -> SqueezeNet:
    """SqueezeNet architecture."""
    backbone = SqueezeNet(version, num_classes=num_classes)
    model = BaseClassifier(backbone)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model


def squeezenet_v0(num_classes: int = 1000, pretrained: bool = False) -> SqueezeNet:
    """
    Constructs a squeezenet_v0 architecture

    Args:
        num_classes (int): The number of classification. Default: 1000.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, output tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> net = squeezenet_v0(1000)

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
    return _squeezenet('squeezenet_v0', '1_0', num_classes=num_classes, pretrained=pretrained)


def squeezenet_v1(num_classes: int = 1000, pretrained: bool = False) -> SqueezeNet:
    """
    Constructs a squeezenet_v1 architecture

    Args:
        num_classes (int): The number of classification. Default: 1000.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, output tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> net = squeezenet_v1(1000)

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
    return _squeezenet('squeezenet_v1', '1_1', num_classes=num_classes, pretrained=pretrained)
