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
""" nonlocal network."""

import math

from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.utils.model_urls import model_urls
from mindvision.msvideo.models.backbones import NLResInflate3D50
from mindvision.msvideo.models.head import DenseHead
from mindvision.msvideo.models.neck import AvgpoolFlatten
from mindvision.utils.load_pretrained_model import LoadPretrainedModel

__all__ = ['non_local']


def non_local(in_d: int = 32,
              in_h: int = 224,
              in_w: int = 224,
              num_classes: int = 400,
              keep_prob: float = 0.5,
              pretrained: bool = False,
              ) -> NLResInflate3D50:
    """
    nonlocal3d model

    Xiaolong Wang.
    "Non-local Neural Networks."
    https://arxiv.org/pdf/1711.07971v3

    Args:
        in_d: Depth of input data, it can be considered as frame number of a video. Default: 32.
        in_h: Height of input frames. Default: 224.
        in_w: Width of input frames. Default: 224.
        num_classes(int): Number of classes, it is the size of classfication score for every sample,
            i.e. :math:`CLASSES_{out}`. Default: 400.
        keep_prob(float): Probability of dropout for multi-dense-layer head, the number of probabilities equals
            the number of dense layers.
        pretrained(bool): If `True`, it will create a pretrained model, the pretrained model will be loaded
            from network. If `False`, it will create a nonlocal3d model with uniform initialization for weight and bias.
    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindvision.msvideo.models import nonlocal3d
        >>>
        >>> net = nonlocal3d()
        >>> x = Tensor(np.random.randn(1, 3, 32, 224, 224).astype(np.float32))
        >>> output = net(x)
        >>> print(output.shape)
        (1, 400)
    """
    last_d = math.ceil(in_d / 32)
    last_h = math.ceil((math.ceil(in_h / 32) + 1) / 4)
    last_w = math.ceil((math.ceil(in_w / 32) + 1) / 4)
    backbone_output_channel = 512 * last_d * last_h * last_w

    backbone = NLResInflate3D50()
    neck = AvgpoolFlatten()
    head = DenseHead(in_channels=backbone_output_channel,
                     num_classes=num_classes,
                     dropout_keep_prob=keep_prob)
    model = BaseClassifier(backbone, neck, head)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        arch = "nonlocal" + "_" + str(in_d) + 'x' + str(in_h) + 'x' + str(in_w)
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model
