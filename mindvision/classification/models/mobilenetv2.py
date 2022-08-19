# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Mobilenet_v2."""

from typing import Optional

from mindspore import nn

from mindvision.classification.models.backbones import MobileNetV2
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.models.head import ConvHead
from mindvision.classification.models.neck import GlobalAvgPooling
from mindvision.classification.models.utils import make_divisible
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel

__all__ = ['mobilenet_v2']


def mobilenet_v2(num_classes: int = 1001,
                 alpha: float = 1.0,
                 round_nearest: int = 8,
                 pretrained: bool = False,
                 resize: int = 224,
                 block: Optional[nn.Cell] = None,
                 norm: Optional[nn.Cell] = None,
                 ) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `MobileNetV2: Inverted Residuals and Linear Bottlenecks <https://arxiv.org/abs/1801.04381>`_.

    Args:
        num_classes (int): The numbers of classes. Default: 1000.
        alpha (float): The channels multiplier for round to 8/16 and others. Default: 1.0.
        round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            set to 1 to turn off rounding. Default is 8.
        pretrained (bool): If True, returns a model pre-trained on IMAGENET. Default: False.
        resize (int): The output size of the resized image. Default: 224.
        block (nn.Cell, optional): Module specifying inverted residual building block for
            mobilenet. Default: None.
        norm (nn.Cell, optional): Norm layer that will be stacked on top of the convoution
            layer. Default: None.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import mobilenet_v2
        >>>
        >>> net = mobilenet_v2()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1001)

    About MobileNetV2:

    The MobileNetV2 architecture is based on an inverted residual structure where the input and output
    of the residual block are thin bottleneck layers opposite to traditional residual models which use
    expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to
    filter features in the intermediate expansion layer.

    Citation:

    .. code-block::

        @article{2018MobileNetV2,
        title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
        author={ Sandler, M.  and  Howard, A.  and  Zhu, M.  and  Zhmoginov, A.  and  Chen, L. C. },
        journal={IEEE},
        year={2018},
        }
    """
    backbone = MobileNetV2(alpha=alpha, round_nearest=round_nearest, block=block, norm=norm)
    neck = GlobalAvgPooling(keep_dims=True)
    inp_channel = make_divisible(1280 * max(1.0, alpha), round_nearest)
    head = ConvHead(input_channel=inp_channel, num_classes=num_classes)
    model = BaseClassifier(backbone, neck, head)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        arch = "mobilenet_v2_" + str(alpha) + "_" + str(resize)
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model
