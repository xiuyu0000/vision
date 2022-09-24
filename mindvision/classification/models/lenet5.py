# Copyright 2021
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
"""LeNet."""

from typing import Optional

from mindvision.classification.models.backbones import LeNet5
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel

__all__ = ['lenet']


def lenet(num_classes: int = 10,
          num_channel: int = 1,
          pretrained: bool = False,
          include_top: bool = True,
          ckpt_file: Optional[str] = None
          ) -> LeNet5:
    """
    Constructs a LeNet architecture from
    `Gradient-based learning applied to document recognition <https://ieeexplore.ieee.org/document/726791>`_.

    Args:
        num_classes (int): The numbers of classes. Default: 1000.
        num_channel (int): The numbers of channels. Default: 1.
        pretrained (bool): If True, returns a model pre-trained on IMAGENET. Default: False.
        include_top (bool): Whether to use the TOP architecture. Default: True.
        ckpt_file (str, optional): The path of checkpoint files. Default: None.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import lenet
        >>>
        >>> net = lenet5()
        >>> x = ms.Tensor(np.ones([1, 1, 32, 32]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 10)

    About LeNet5:

    LeNet5 trained with the back-propagation algorithm constitute the best example of
    a successful gradient based learning technique. Given an appropriate network architecture,
    gradient-based learning algorithms can be used to synthesize a complex decision surface
    that can classify high-dimensional patterns, such as handwritten characters, with
    minimal preprocessing.

    Citation:

    .. code-block::

        @article{1998Gradient,
          title={Gradient-based learning applied to document recognition},
          author={ Lecun, Y.  and  Bottou, L. },
          journal={Proceedings of the IEEE},
          volume={86},
          number={11},
          pages={2278-2324},
          year={1998}
        }
    """
    backbone = LeNet5(num_classes=num_classes, num_channel=num_channel, include_top=include_top)
    model = BaseClassifier(backbone)

    if pretrained and not ckpt_file:
        # Download the pre-trained checkpoint file from url, and load checkpoint file.
        arch = "lenet"
        LoadPretrainedModel(model, model_urls[arch]).run()
    elif ckpt_file:
        # Just load checkpoint file.
        LoadPretrainedModel(model, ckpt_file).run()

    return model
