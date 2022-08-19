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
"""inception block."""

from mindspore import nn
import mindspore.ops as ops


class Inception(nn.Cell):
    """
    Args:
        num_channels (int): input channel number
        c1 (int): out_channels of 1*1 conv layer
        c2 (list[int]): parameter of 3*3 conv layer
        c3 (list[int]): parameter of 5*5 conv layer
        c4 (int): out_channels of maxpool layer

    Returns:
        Tensor, output tensor.
    """
    def __init__(self, num_channels: int, c1: int, c2, c3, c4: int):
        super(Inception, self).__init__()
        #1*1 conv
        self.conv1 = nn.Conv2d(num_channels, c1, kernel_size=1)
        #3*3 conv
        self.conv2 = nn.SequentialCell([
            nn.Conv2d(num_channels, c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3),
            nn.ReLU()
        ])
        #5*5 conv
        self.conv3 = nn.SequentialCell([
            nn.Conv2d(num_channels, c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3),
            nn.ReLU()
        ])
        #3*3 maxpool, 1*1 conv
        self.conv4 = nn.SequentialCell([
            nn.MaxPool2d(kernel_size=3, stride=1, pad_mode='same'),
            nn.Conv2d(num_channels, c4, kernel_size=1),
            nn.ReLU()
        ])

    def construct(self, x):
        op = ops.Concat(axis=1)
        return op((self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)))
