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
"""PointNet2 Classfication Backbone"""

from mindspore import nn
from mindspore import ops

from mindvision.ms3d.models.blocks.pointnet2_sa import PointNet2SetAbstraction
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ['PointNet2ClsBackbone']


@ClassFactory.register(ModuleType.BACKBONE)
class PointNet2ClsBackbone(nn.Cell):
    """
    PointNet2 Classfication architecture.
    Args:
            normal_channel(bool): Whether to use the channels of points' normal vector. Default: True.
    Inputs:
            - points(Tensor) - Tensor of original points. shape:[batch, channels, npoints].
    Outputs:
            Tensor of shape :[batch, 1024].

    Supported Platforms:
            ``GPU``

    Examples:
        >> import numpy as np
        >> import mindspore as ms
        >> from mindspore import Tensor, context
        >> from mindvision.ms3d.models.backbones import pointnet2cls
        >> context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, max_call_depth=2000)
        >> net = PointNet2ClsBackbone(normal_channel=False)
        >> xyz = Tensor(np.ones((24,6, 1024)),ms.float32)
        >> output = net(xyz)
        >> print(output.shape)
        (24, 1024)

    About PointNet2ClsBackbone:
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

    def __init__(self, normal_channel=False):
        super(PointNet2ClsBackbone, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        # 512 npoint = points sampled in farthest point sampling
        # 0.2 radius = search radius in local region
        # 32 nsample = how many points in each local region
        # [64,64,128] mlp = output size for MLP on each point
        # + 3 = xyz 3-dim coordinates
        self.sa1 = PointNet2SetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],
                                           group_all=False)
        self.sa2 = PointNet2SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                           group_all=False)
        self.sa3 = PointNet2SetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                           mlp=[256, 512, 1024], group_all=True)

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, xyz):
        """construct method"""
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        x = self.reshape(l3_points, (-1, 1024))
        return x
