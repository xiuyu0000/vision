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
"""SeqMLP module"""
from mindspore import nn
from mindspore import ops


class SeqMLP(nn.Cell):
    """
    SeqMLP module.
    The input data is x(Tensor): shape(B, C, N) or shape(B, C, N, K),
    where C is the number of channel, N and K is the number of points.

    Args:
        in_channel (int): Channel number of input data.
            Default: 3.
        num_channel (tuple[int]): Channel number for a series of MLP layers.
            Default: (64, 128, 1024).

    Returns:
        Tensor: shape(B, C, N)

    Examples:
        >>> data = Tensor(np.random.randn(8, 3, 1000), dtype=mindspore.float32)
        >>> model = SeqMLP(in_channel=3, num_channel=(64, 128, 1024))

        >>> predict = model(data)
        >>> print(predict.shape)

    """

    def __init__(self, in_channel=3, num_channel=(64, 128, 1024)):
        super(SeqMLP, self).__init__()
        self.modules = nn.SequentialCell()
        for n in num_channel:
            self.modules.append(nn.Conv2d(in_channel, n, 1, has_bias=True, bias_init='normal'))
            self.modules.append(nn.BatchNorm2d(n))
            self.modules.append(nn.ReLU())
            in_channel = n

    def construct(self, x):
        """SeqMLP construct"""
        in_dim = len(x.shape)
        if in_dim == 3:
            x = ops.ExpandDims()(x, -1)
        x = self.modules(x)
        if in_dim == 3:
            x = ops.Squeeze(-1)(x)

        return x
