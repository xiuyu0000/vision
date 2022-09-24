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
""" ARN network."""

from typing import Optional

from mindspore import nn
from mindspore import ops as P

from mindvision.msvideo.models.head.arn_head import ARNHead
from mindvision.utils.load_pretrained_model import LoadPretrainedModel
from mindvision.classification.utils.model_urls import model_urls
from mindvision.msvideo.models.backbones.arn import ARNBackbone
from mindvision.msvideo.models.base import BaseRecognizer
from mindvision.msvideo.models.embedding.arn_embed import ARNEmbedding
from mindvision.msvideo.models.neck.arn_neck import ARNNeck

__all__ = ['arn']


def arn(support_num_per_class: int = 5,
        query_num_per_class: int = 3,
        class_num: int = 5,
        is_c3d: bool = False,
        in_channels: Optional[int] = 3,
        out_channels: Optional[int] = 64,
        pooling: Optional[nn.Cell] = P.MaxPool3D(
            kernel_size=2, strides=2),
        jigsaw: int = 10,
        sigma: int = 100,
        pretrained: bool = False,
        arch: Optional[str] = None
        ) -> nn.Cell:
    """
    Constructs a ARN architecture from
    `Few-shot Action Recognition via Permutation-invariant Attention <https://arxiv.org/pdf/2001.03905.pdf>`.

    Args:
        support_num_per_class (int): Number of samples in support set per class. Default: 5.
        query_num_per_class (int): Number of samples in query set per class. Default: 3.
        class_num (int): Number of classes. Default: 5.
        is_c3d (bool): Specifies whether the network uses C3D as embendding for ARN. Default: False.
        in_channels: The number of channels of the input feature. Default: 3.
        out_channels: The number of channels of the output of hidden layers (only used when is_c3d is set to False).
            Default: 64.
        pooling: The definition of pooling layer (only used when is_c3d is set to False).
            Default: ops.MaxPool3D(kernel_size=2, strides=2).
        jigsaw (int): Number of the output dimension for spacial-temporal jigsaw discriminator. Default: 10.
        sigma: Controls the slope of PN. Default: 100.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        arch (str | None, optional): Pre-trained model's architecture name. Default: None.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(E, N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(CLASSES_NUM, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.msvideo.models import arn
        >>>
        >>> net = arn(5, 3, 5, False, 3, 64, ops.MaxPool3D(kernel_size=2, strides=2), 10, 100)
        >>> x = ms.Tensor(np.random.randn(1, 10, 3, 16, 128, 128), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (5, 5)

    About ARN:

    TODO: ARN introduction.

    Citation:

    .. code-block::

        @article{zhang2020few,
            title={Few-shot Action Recognition with Permutation-invariant Attention},
            author={Zhang, Hongguang and Zhang, Li and Qi, Xiaojuan and Li, Hongdong and Torr, Philip HS
                and Koniusz, Piotr},
            journal={arXiv preprint arXiv:2001.03905},
            year={2020}
        }
    """

    embedding = ARNEmbedding(support_num_per_class=support_num_per_class,
                             query_num_per_class=query_num_per_class,
                             class_num=class_num,
                             is_c3d=is_c3d,
                             in_channels=in_channels,
                             out_channels=out_channels,
                             pooling=pooling
                             )
    backbone = ARNBackbone(jigsaw=jigsaw,
                           support_num_per_class=support_num_per_class,
                           query_num_per_class=query_num_per_class,
                           class_num=class_num
                           )
    neck = ARNNeck(class_num=class_num,
                   support_num_per_class=support_num_per_class,
                   sigma=sigma)
    head = ARNHead(class_num=class_num,
                   query_num_per_class=query_num_per_class
                   )
    model = BaseRecognizer(
        backbone=backbone, embedding=embedding, neck=neck, head=head)
    if pretrained:
        # Download the pre-trained checkpoint file from url, and load ckpt file.
        # TODO: model_urls is not defined yet.
        LoadPretrainedModel(model, model_urls[arch]).run()
    return model
