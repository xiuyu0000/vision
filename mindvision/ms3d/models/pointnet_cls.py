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
"""PointNet Classification."""

from mindvision.ms3d.models.backbones.pointnet import PointNet
from mindvision.ms3d.models.head.cls_head import ClsHead
from mindvision.classification.models.classifiers import BaseClassifier

__all__ = ['pointnet_cls']


def pointnet_cls(global_feat: bool = True,
                 feature_transform: bool = True,
                 pretrained: bool = False
                 ) -> PointNet:
    """
    Constructs a Pointnet classification architecture from
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation <https://arxiv.org/abs/1612.00593>.

    Args:
        global_feat(bool): Choose task type, classification(True) or segmentation(False). Default: True.
        feature_transform(bool): Whether to use feature transform. Default: True.
        pretrained (bool): If True, returns a model pre-trained on IMAGENET. Default: False.

    Inputs:
        - points(Tensor) - Tensor of original points. shape:[batch, channels, npoints].

    Outputs:
        Tensor of shape :[batch, 40].

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindvision.ms3d.models.pointnet_cls import pointnet_cls
        >>> net = pointnet_cls()
        >>> x = ms.Tensor(np.ones((32, 3, 1024)), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (32, 40)

    About PointNet:

    PointNet is a hierarchical neural network that applies PointNet recursively on a nested partitioning
    of the input point set. The author of this paper proposes a method of applying deep learning model directly
    to point cloud data, which is called pointnet.

    Citation:

    .. code-block::

        @article{qi2016pointnet,
        title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
        author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
        journal={arXiv preprint arXiv:1612.00593},
        year={2016}
        }
    """
    backbone = PointNet(global_feat=global_feat, feature_transform=feature_transform)
    head = ClsHead(input_channel=1024, num_classes=40, mid_channel=[512, 256], keep_prob=0.7)
    model = BaseClassifier(backbone, neck=None, head=head)

    if pretrained:
        raise ValueError("pretrained is not supported.")
    return model
