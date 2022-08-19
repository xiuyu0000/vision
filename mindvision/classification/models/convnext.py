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
"""ConvNext"""
from typing import List

from mindvision.classification.models.backbones import ConvNeXt
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.models.head import DenseHead
from mindvision.classification.models.neck import AvgPoolingLayerNorm
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel

__all__ = [
    'convnext_tiny',
    'convnext_small',
    'convnext_base',
    'convnext_large',
    'convnext_xlarge'
]


def _convnext(arch: str,
              depths: List[int],
              dims: List[int],
              pretrained: bool,
              in_channels: int = 3,
              num_classes: int = 1000,
              drop_path_rate: float = 0.,
              layer_scale: float = 1e-6) -> ConvNeXt:
    """ConvNext architecture."""
    backbone = ConvNeXt(
        in_channels=in_channels,
        depths=depths,
        dims=dims,
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale
    )
    neck = AvgPoolingLayerNorm(num_channels=dims[-1])
    head = DenseHead(input_channel=dims[-1], num_classes=num_classes)
    model = BaseClassifier(backbone, neck, head)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model


def convnext_tiny(pretrained: bool = False,
                  in_channels: int = 3,
                  num_classes: int = 1000,
                  drop_path_rate: float = 0.,
                  layer_scale: float = 1e-6
                  ) -> ConvNeXt:
    """
    Constructs a ConvNext-tiny architecture.

    Args:
        pretrained(bool): Whether to download and load the pre-trained model. Default: False.
        in_channels(int): Number of input channels.
        num_classes(int): The number of classification. Default: 1000.
        drop_path_rate(float): Stochastic depth rate. Default: 0.
        layer_scale(float): Init value for Layer Scale. Default: 1e-6.

    Inputs:
        - **x**(Tensor) - Tensor of shape: math: `(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape: math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import convnext_tiny
        >>>
        >>> net = convnext_tiny()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>>print(output.shape)
        (1, 1000)

    About ConvNext:

    Convnext pure convolutional neural network is proposed, which is aimed at the very popular swing transformer
    in 2021. Through a series of experimental comparisons, convnext has faster reasoning speed and higher accuracy
    than swing transformer under the same flops.

    Citation:

    .. code-block::

        @article{,
        title={A ConvNet for the 2020s},
        author={Zhuang, Liu. and Hanzi, Mao. and Chao-Yuan, Wu.},
        journal={},
        year={}
        }
    """
    return _convnext(
        "convnext_tiny", [3, 3, 9, 3], [96, 192, 384, 768],
        pretrained, in_channels, num_classes, drop_path_rate, layer_scale)


def convnext_small(pretrained: bool = False,
                   in_channels: int = 3,
                   num_classes: int = 1000,
                   drop_path_rate: float = 0.,
                   layer_scale: float = 1e-6
                   ) -> ConvNeXt:
    """
    Constructs a ConvNext-small architecture.

    Args:
        pretrained(bool): Whether to download and load the pre-trained model. Default: False.
        in_channels(int): Number of input channels.
        num_classes(int): The number of classification. Default: 1000.
        drop_path_rate(float): Stochastic depth rate. Default: 0.
        layer_scale(float): Init value for Layer Scale. Default: 1e-6.

    Inputs:
        - **x**(Tensor) - Tensor of shape: math: `(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape: math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import convnext_small
        >>>
        >>> net = convnext_small()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>>print(output.shape)
        (1, 1000)

    About ConvNext:

    Convnext pure convolutional neural network is proposed, which is aimed at the very popular swing transformer
    in 2021. Through a series of experimental comparisons, convnext has faster reasoning speed and higher accuracy
    than swing transformer under the same flops.

    Citation:

    .. code-block::

        @article{,
        title={A ConvNet for the 2020s},
        author={Zhuang, Liu. and Hanzi, Mao. and Chao-Yuan, Wu.},
        journal={},
        year={}
        }
    """
    return _convnext(
        "convnext_small", [3, 3, 27, 3], [96, 192, 384, 768],
        pretrained, in_channels, num_classes, drop_path_rate, layer_scale)


def convnext_base(pretrained: bool = False,
                  in_channels: int = 3,
                  num_classes: int = 1000,
                  drop_path_rate: float = 0.,
                  layer_scale: float = 1e-6
                  ) -> ConvNeXt:
    """
    Constructs a ConvNext-base architecture.

    Args:
        pretrained(bool): Whether to download and load the pre-trained model. Default: False.
        in_channels(int): Number of input channels.
        num_classes(int): The number of classification. Default: 1000.
        drop_path_rate(float): Stochastic depth rate. Default: 0.
        layer_scale(float): Init value for Layer Scale. Default: 1e-6.

    Inputs:
        - **x**(Tensor) - Tensor of shape: math: `(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape: math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import convnext_base
        >>>
        >>> net = convnext_base()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>>print(output.shape)
        (1, 1000)

    About ConvNext:

    Convnext pure convolutional neural network is proposed, which is aimed at the very popular swing transformer
    in 2021. Through a series of experimental comparisons, convnext has faster reasoning speed and higher accuracy
    than swing transformer under the same flops.

    Citation:

    .. code-block::

        @article{,
        title={A ConvNet for the 2020s},
        author={Zhuang, Liu. and Hanzi, Mao. and Chao-Yuan, Wu.},
        journal={},
        year={}
        }
    """
    return _convnext(
        "convnext_base", [3, 3, 27, 3], [128, 256, 512, 1024],
        pretrained, in_channels, num_classes, drop_path_rate, layer_scale)


def convnext_large(pretrained: bool = False,
                   in_channels: int = 3,
                   num_classes: int = 1000,
                   drop_path_rate: float = 0.,
                   layer_scale: float = 1e-6
                   ) -> ConvNeXt:
    """
    Constructs a ConvNext-large architecture.

    Args:
        pretrained(bool): Whether to download and load the pre-trained model. Default: False.
        in_channels(int): Number of input channels.
        num_classes(int): The number of classification. Default: 1000.
        drop_path_rate(float): Stochastic depth rate. Default: 0.
        layer_scale(float): Init value for Layer Scale. Default: 1e-6.

    Inputs:
        - **x**(Tensor) - Tensor of shape: math: `(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape: math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import convnext_large
        >>>
        >>> net = convnext_large()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>>print(output.shape)
        (1, 1000)

    About ConvNext:

    Convnext pure convolutional neural network is proposed, which is aimed at the very popular swing transformer
    in 2021. Through a series of experimental comparisons, convnext has faster reasoning speed and higher accuracy
    than swing transformer under the same flops.

    Citation:

    .. code-block::

        @article{,
        title={A ConvNet for the 2020s},
        author={Zhuang, Liu. and Hanzi, Mao. and Chao-Yuan, Wu.},
        journal={},
        year={}
        }
    """
    return _convnext(
        "convnext_large", [3, 3, 27, 3], [192, 384, 768, 1536],
        pretrained, in_channels, num_classes, drop_path_rate, layer_scale)


def convnext_xlarge(pretrained: bool = False,
                    in_channels: int = 3,
                    num_classes: int = 1000,
                    drop_path_rate: float = 0.,
                    layer_scale: float = 1e-6
                    ) -> ConvNeXt:
    """
    Constructs a ConvNext-xlarge architecture.

    Args:
        pretrained(bool): Whether to download and load the pre-trained model. Default: False.
        in_channels(int): Number of input channels.
        num_classes(int): The number of classification. Default: 1000.
        drop_path_rate(float): Stochastic depth rate. Default: 0.
        layer_scale(float): Init value for Layer Scale. Default: 1e-6.

    Inputs:
        - **x**(Tensor) - Tensor of shape: math: `(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape: math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import convnext_xlarge
        >>>
        >>> net = convnext_xlarge()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>>print(output.shape)
        (1, 1000)

    About ConvNext:

    Convnext pure convolutional neural network is proposed, which is aimed at the very popular swing transformer
    in 2021. Through a series of experimental comparisons, convnext has faster reasoning speed and higher accuracy
    than swing transformer under the same flops.

    Citation:

    .. code-block::

        @article{,
        title={A ConvNet for the 2020s},
        author={Zhuang, Liu. and Hanzi, Mao. and Chao-Yuan, Wu.},
        journal={},
        year={}
        }
    """
    return _convnext(
        "convnext_xlarge", [3, 3, 27, 3], [256, 512, 1024, 2048],
        pretrained, in_channels, num_classes, drop_path_rate, layer_scale)
