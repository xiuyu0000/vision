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
"""PointNet Backbone."""

from mindspore import nn
from mindspore import ops

from mindvision.ms3d.models.blocks.t_net import STN3D, STNkD
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["PointNet"]


@ClassFactory.register(ModuleType.BACKBONE)
class PointNet(nn.Cell):
    """
    PointNet Backbone.

    Args:
        global_feat(bool): Choose task type, classification(True) or segmentation(False). Default: True.
        feature_transform(bool): Whether to use feature transform. Default: True.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(B, 3, N)`.

    Outputs:
        Tensor of shape :math:`(B, 1024)`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>>data = (32, 3, 1024)
        >>>uniformreal = ops.UniformReal()
        >>>sim_data = uniformreal(data)
        >>>pointfeat = PointNet(global_feat=True)
        >>>out, _, _, _ = pointfeat(sim_data)
        >>>print('global feat', out.shape)
        (32, 1024)

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

    def __init__(self, global_feat=True, feature_transform=True):
        super(PointNet, self).__init__()
        self.stn = STN3D()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.transpose = ops.Transpose()
        self.batmatmul = ops.BatchMatMul()
        self.argmaxwithvalue = ops.ArgMaxWithValue(axis=2, keep_dims=True)
        self.reshape = ops.Reshape()

        self.relu = ops.ReLU()
        self.tile = ops.Tile()
        self.cat = ops.Concat(axis=1)

        if self.feature_transform:
            self.fstn = STNkD(k=64)

    def construct(self, x):
        """
        PointNet construct.
        """
        transf = self.stn(x)
        x = self.transpose(x, (0, 2, 1))
        x = self.batmatmul(x, transf)
        x = self.transpose(x, (0, 2, 1))
        x = self.relu(ops.Squeeze(-1)(self.bn1(ops.ExpandDims()(self.conv1(x), -1))))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = self.transpose(x, (0, 2, 1))
            x = self.batmatmul(x, trans_feat)
            x = self.transpose(x, (0, 2, 1))
        else:
            trans_feat = None

        pointfeat = x
        x = self.relu(ops.Squeeze(-1)(self.bn2(ops.ExpandDims()(self.conv2(x), -1))))
        x = ops.Squeeze(-1)(self.bn3(ops.ExpandDims()(self.conv3(x), -1)))
        x = self.argmaxwithvalue(x)[1]

        if self.global_feat:
            x = self.reshape(x, (-1, 1024))
        return x, transf, trans_feat, pointfeat
