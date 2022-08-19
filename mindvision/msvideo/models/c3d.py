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
""" C3D network."""

import math
from typing import List, Tuple, Union

from mindspore import nn
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.models.head import MultilayerDenseHead
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel
from mindvision.msvideo.models.backbones import C3D

__all__ = ['c3d']


def c3d(in_d: int = 16,
        in_h: int = 112,
        in_w: int = 112,
        in_channel: int = 3,
        kernel_size: Union[int, List[int], Tuple[int]] = (3, 3, 3),
        head_channel: Union[List[int], Tuple[int]] = (4096, 4096),
        num_classes: int = 400,
        keep_prob: Union[List[int], Tuple[int]] = (0.5, 0.5, 1.0),
        pretrained: bool = False,
        ) -> C3D:
    """
    TODO: introduction c3d network.

    Args:
        in_d: Depth of input data, it can be considered as frame number of a video. Default: 16.
        in_h: Height of input frames. Default: 112.
        in_w: Width of input frames. Default: 112.
        in_channel(int): Number of channel of input data. Default: 3.
        kernel_size(Union[int, List[int], Tuple[int]]): Kernel size for every conv3d layer in C3D.
            Default: (3, 3, 3).
        head_channel(List[int]): Hidden size of multi-dense-layer head. Default: [4096, 4096].
        num_classes(int): Number of classes, it is the size of classfication score for every sample,
            i.e. :math:`CLASSES_{out}`. Default: 400.
        keep_prob(List[int]): Probability of dropout for multi-dense-layer head, the number of probabilities equals
            the number of dense layers.
        pretrained(bool): If `True`, it will create a pretrained model, the pretrained model will be loaded
            from network. If `False`, it will create a c3d model with uniform initialization for weight and bias.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindvision.msvideo.models import c3d
        >>>
        >>> net = c3d(16, 128, 128)
        >>> x = ms.Tensor(np.ones([1, 3, 16, 128, 128]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 400)

    About c3d:

    TODO: c3d introduction.

    Citation:

    .. code-block::

        TODO: c3d Citation.
    """
    last_d = math.ceil(in_d / 16)
    last_h = math.ceil((math.ceil(in_h / 16) + 1) / 2)
    last_w = math.ceil((math.ceil(in_w / 16) + 1) / 2)
    backbone_output_channel = 512 * last_d * last_h * last_w
    head_channel = list(head_channel)

    backbone = C3D(in_channel=in_channel,
                   kernel_size=kernel_size)
    neck = nn.Flatten()
    head = MultilayerDenseHead(input_channel=backbone_output_channel,
                               mid_channel=head_channel,
                               num_classes=num_classes,
                               activation=['relu', 'relu', None],
                               keep_prob=keep_prob)
    model = BaseClassifier(backbone, neck, head)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load checkpoint file.
        arch = "C3D" + "_" + str(in_h) + 'x' + str(in_w)
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model
