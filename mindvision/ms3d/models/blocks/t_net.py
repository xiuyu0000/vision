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
"""T-Net module"""

import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor


class STN3D(nn.Cell):
    """
    Input transform module. The input data is x(Tensor):shape(B,3,N),
    where B is the batch size and N is the number of points.

    Args:
        in_channel(int):Channel number of input data.
            Default:3.

    Returns:
        Tensor:shape(B,3,3)

    Example:
        >>>shape1 = (32, 3, 2500)
        >>>uniformreal = mindspore.ops.UniformReal()
        >>>sim_data = uniformreal(shape1)
        >>>trans = STN3D()
        >>>out = trans(sim_data)
        >>>print('stn3d', out.shape)

    """

    def __init__(self, in_channel=3):
        super(STN3D, self).__init__()
        self.conv1 = mindspore.nn.Conv1d(in_channel, 64, 1, has_bias=True, bias_init='normal')
        self.conv2 = mindspore.nn.Conv1d(64, 128, 1, has_bias=True, bias_init='normal')
        self.conv3 = mindspore.nn.Conv1d(128, 1024, 1, has_bias=True, bias_init='normal')
        self.fc1 = nn.Dense(1024, 512)
        self.fc2 = nn.Dense(512, 256)
        self.fc3 = nn.Dense(256, 9)
        self.relu = ops.ReLU()

        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.argmaxwithvalue = ops.ArgMaxWithValue(axis=2, keep_dims=True)
        self.s1 = Tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], mindspore.float32)

    def construct(self, x):
        """Input transform construct"""
        batchsize = x.shape[0]

        x = self.relu(ops.Squeeze(-1)(self.bn1(ops.ExpandDims()(self.conv1(x), -1))))
        x = self.relu(ops.Squeeze(-1)(self.bn2(ops.ExpandDims()(self.conv2(x), -1))))
        x = self.relu(ops.Squeeze(-1)(self.bn3(ops.ExpandDims()(self.conv3(x), -1))))
        x = self.argmaxwithvalue(x)[1]

        x = self.reshape(x, (-1, 1024))

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        multiples = (batchsize, 1)

        iden = self.tile(self.s1.view(1, 9), multiples)

        x = x + iden
        x = self.reshape(x, (-1, 3, 3))
        return x


class STNkD(nn.Cell):
    """
    Feature transform module.
    The input data is x(Tensor):shape(B,k,N),
    where B is the batch size ,k is the dimension of the transform matrix
    and N is the number of points.

    Args:
        k(int):Dimension of the feature transform matrix.
            Default:64.

    Returns:
        Tensor:shape(B,k,k)

    Example:
        >>>shape2 = (32, 64, 2500)
        >>>uniformreal = mindspore.ops.UniformReal()
        >>>sim_data_64d = uniformreal(shape2)
        >>>trans = STNkD(k=64)
        >>>out = trans(sim_data_64d)
        >>>print('stn64d', out.shape)

    """

    def __init__(self, k=64):
        super(STNkD, self).__init__()
        self.conv1 = mindspore.nn.Conv1d(k, 64, 1, has_bias=True, bias_init='normal')
        self.conv2 = mindspore.nn.Conv1d(64, 128, 1, has_bias=True, bias_init='normal')
        self.conv3 = mindspore.nn.Conv1d(128, 1024, 1, has_bias=True, bias_init='normal')
        self.fc1 = nn.Dense(1024, 512)
        self.fc2 = nn.Dense(512, 256)
        self.fc3 = nn.Dense(256, k * k)
        self.relu = ops.ReLU()

        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.argmaxwithvalue = ops.ArgMaxWithValue(axis=2, keep_dims=True)

        self.k = k
        self.s2 = ops.Eye()(self.k, self.k, mindspore.float32).flatten()

    def construct(self, x):
        """Feature transform construct"""
        batchsize = x.shape[0]

        x = self.relu(ops.Squeeze(-1)(self.bn1(ops.ExpandDims()(self.conv1(x), -1))))
        x = self.relu(ops.Squeeze(-1)(self.bn2(ops.ExpandDims()(self.conv2(x), -1))))
        x = self.relu(ops.Squeeze(-1)(self.bn3(ops.ExpandDims()(self.conv3(x), -1))))
        x = self.argmaxwithvalue(x)[1]

        x = self.reshape(x, (-1, 1024))

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        tile = ops.Tile()
        multiples = (batchsize, 1)

        iden = tile(self.s2.view(1, self.k * self.k), multiples)
        x = x + iden
        x = self.reshape(x, (-1, self.k, self.k))

        return x
