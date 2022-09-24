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
"""PointNet2Model"""

from mindvision.ms3d.models.backbones.pointnet2clsbackbone import PointNet2ClsBackbone
from mindvision.ms3d.models.base import BaseRecognizer
from mindvision.ms3d.models.head.point2_head import Point2ClsHead

__all__ = ['Pointnet2clsModel']


def Pointnet2clsModel(normal_channel: bool = False, pretrained: bool = False):
    """
    Constructs a PointnetNet2 architecture from
    PointnetNet2: Deep Hierarchical Feature Learning on Point Sets in a Metric Space <https://arxiv.org/abs/1706.02413>.

    Args:
        normal_channel (bool): Whether to use the channels of points' normal vector. Default: True.
        pretrained (bool): If True, returns a model pre-trained on IMAGENET. Default: False.

    Inputs:
        - points(Tensor) - Tensor of original points. shape:[batch, channels, npoints].

    Outputs:
        Tensor of shape :[batch, 40].

    Supported Platforms:
        ``GPU``

    Examples:
        >> import numpy as np
        >> import mindspore as ms
        >> from mindspore import Tensor, context
        >> from mindvision.ms3d.models.backbones import pointnet2cls
        >> context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, max_call_depth=2000)
        >> net = Pointnet2clsModel(normal_channel=True, pretrained:bool=False)
        >> xyz = Tensor(np.ones((24,6, 1024)),ms.float32)
        >> output = net(xyz)
        >> print(output.shape)
        (24, 40)

    About PointNet2Cls:

    This architecture is based on PointNet2 classfication SSG,
    compared with PointNet, PointNet2_SSG added local feature extraction.

    Citation

    .. code-block::

        @article{qi2017pointnetplusplus,
          title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
          author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
          journal={arXiv preprint arXiv:1706.02413},
          year={2017}
        }
    """
    backbone = PointNet2ClsBackbone(normal_channel=normal_channel)
    head = Point2ClsHead(input_channel=1024, num_classes=40, mid_channel=[512, 256], keep_prob=0.4)
    model = BaseRecognizer(backbone, neck=None, head=head)

    if pretrained:
        raise ValueError("pretrained is not supported.")
    return model
