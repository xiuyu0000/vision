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
"""PointNetDense Backbone."""

import mindspore.ops as ops
from mindspore import nn
from mindvision.ms3d.models.backbones.pointnet import PointNet
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["PointNetDense"]


@ClassFactory.register(ModuleType.BACKBONE)
class PointNetDense(nn.Cell):
    """
    PointNetDense Backbone.

    Args:
        k(int): The number of classes. Default: 50.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(B, 3, N)`.

    Outputs:
        Tensor of shape :math:`(B, N, k)`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> data = (32, 3, 1024)
        >>> uniformreal = ops.UniformReal()
        >>> sim_data = uniformreal(data)
        >>> seg = PointNetDense(k=50)
        >>> out = seg(sim_data)
        >>> print('seg', out.shape)
        (32, 1024, 50)

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

    def __init__(self, k=50):
        super(PointNetDense, self).__init__()
        self.k = k
        self.feat = PointNet(global_feat=False, feature_transform=False)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)

        self.logsoftmax = nn.LogSoftmax(axis=-1)
        self.relu = ops.ReLU()
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.cat = ops.Concat(axis=1)

    def construct(self, x):
        """
        PointNetDense construct.
        """
        batchsize = x.shape[0]
        n_pts = x.shape[2]
        x, _, _, pointfeat = self.feat(x)

        multiples = (1, 1, n_pts)
        x = self.tile(self.reshape(x, (-1, 1024, 1)), multiples)
        x = self.cat((x, pointfeat))
        x = self.relu(ops.Squeeze(-1)(self.bn1(ops.ExpandDims()(self.conv1(x), -1))))
        x = self.relu(ops.Squeeze(-1)(self.bn2(ops.ExpandDims()(self.conv2(x), -1))))
        x = self.relu(ops.Squeeze(-1)(self.bn3(ops.ExpandDims()(self.conv3(x), -1))))
        x = self.conv4(x)
        transpose = ops.Transpose()
        x = transpose(x, (0, 2, 1))
        x = self.logsoftmax(x.view(-1, self.k))
        x = x.view(batchsize, n_pts, self.k)

        return x
