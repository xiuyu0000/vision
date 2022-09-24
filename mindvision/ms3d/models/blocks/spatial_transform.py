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
"""KNN module"""
from Model.Knn import KNN
import mindspore.numpy as np
from mindspore import ops
from mindspore import nn


class Getgraphfeature():
    """
        Get Graph Feature Module

        The input data is x shape(B,C,N)
        where B is the batch size ,C is the dimension of the transform matrix,
         N is the number of points,and K is k-NearestNeighbor Parameter

        :param x:   input data
        :param k:   k-NearestNeighbor Parameter
        :param idx: Return value in KNN
        :return: Tensor shape  (B,C,N,K)
    """

    def __init__(self, x, k=20, idx=None):
        self.x = x
        self.k = k
        self.idx = idx

    def func(self):
        """Get Graph Feature Func"""
        batch_size = self.x.shape[0]
        num_points = self.x.shape[2]
        x = self.x.view(batch_size, -1, num_points)
        knn = KNN(x, k=self.k)
        if self.idx is None:
            idx = knn.func()
        idx_base = np.arange(0, batch_size).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.shape[-1]
        _, num_dims, _ = x.shape
        x = x.transpose(0, 2, 1)  # 2048 5
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.resize(batch_size, num_points, self.k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims)  # .repeat(1, 1, k, 1)
        x = np.repeat(x, self.k, axis=2)
        op = ops.Concat(3)
        feature = op((feature - x, x)).transpose(0, 3, 1, 2)  # .permute(0,3,1,2)

        return feature


class TransformNet(nn.Cell):
    """
    Transform-Net module

    The input data is x(Tensor):shape(B,3,N),
    where B is the batch size and N is the number of points.
    """

    def __init__(self, args):
        super(TransformNet, self).__init__()
        self.args = args
        self.k = 3
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv1 = nn.SequentialCell(nn.Conv2d(6, 64, kernel_size=1, bias_init=False),
                                       self.bn1,
                                       nn.LeakyReLU(0.2))
        self.conv2 = nn.SequentialCell(nn.Conv2d(64, 128, kernel_size=1, bias_init=False),
                                       self.bn2,
                                       nn.LeakyReLU(0.2))
        self.conv3 = nn.SequentialCell(nn.Conv1d(128, 1024, kernel_size=1, bias_init=False),
                                       self.bn3,
                                       nn.LeakyReLU(0.2))
        self.linear1 = nn.Dense(1024, 512, bias_init=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.linear2 = nn.Dense(512, 256, bias_init=False)
        self.bn5 = nn.BatchNorm1d(256)

        self.transform = nn.Dense(256, 3 * 3)

    def construct(self, x):
        """TransformNet Construct"""
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.max(dim=-1, keepdim=False)[0]
        x = self.conv3(x)
        x = x.max(dim=-1, keepdim=False)[0]
        leaky_relu = nn.LeakyReLU(0.2)
        x = leaky_relu(self.bn3(self.linear1(x)))
        x = leaky_relu(self.bn4(self.linear2(x)))
        x = self.transform(x)
        x = x.view(batch_size, 3, 3)
        return x


class SpatialTransform(nn.Cell):
    '''
    SpatialTransform Moule

    The input data is x(Tensor):shape(B,N,3),
    where B is the batch size and N is the number of points.
    '''

    def __init__(self, args):
        super(SpatialTransform, self).__init__()
        self.args = args
        self.k = args.k
        self.transform_net = TransformNet(args)

    def construct(self, x):
        """SpatialTransform Construct"""
        get_graph_feature = Getgraphfeature(x, k=self.k)
        x0 = get_graph_feature.func()
        t = self.transform_net(x0)
        x = x.tranpose(0, 2, 1)
        x = ops.BatchMatMul(x, t)
        x = x.transpose(0, 1, 2)
        return x
