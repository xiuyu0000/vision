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
"""Edge Conv module"""

from collections import OrderedDict
from Model.SpatialTransform import Getgraphfeature

from mindspore import nn


class EdgeConv(nn.Cell):
    """
    Edge Conv Module. The input data is x(Tensor):shape(B,F),
    where B is the batch size and F is the dimension of feature.
    """

    def __init__(self, layers, k=20):
        super(EdgeConv, self).__init__()
        self.k = k
        self.layers = layers
        if layers is not None:
            self.mlp = None
        else:
            mlp_layers = OrderedDict()
            for i in range(len(self.layers) - 1):
                if i == 0:
                    mlp_layers['conv_bn_block_{}'.format(i + 1)] = self.conv_bn_block(2 * self.layers[i],
                                                                                      self.layers[i + 1],
                                                                                      1)
                else:
                    mlp_layers['conv_bn_block_{}'.format(i + 1)] = self.conv_bn_block(self.layers[i],
                                                                                      self.layers[i + 1], 1)
            self.mlp = nn.SequentialCell(mlp_layers)

    def conv_bn_block(self, c_input, c_output, kernel_size):
        return nn.SequentialCell(
            nn.Conv2d(c_input, c_output, kernel_size),
            nn.BatchNorm1d(c_output),
            nn.LeakyReLU(0.2)
        )

    def fc_bn_block(self, c_input, c_output):
        return nn.SequentialCell(
            nn.Dense(c_input, c_output),
            nn.BatchNorm1d(c_output),
            nn.LeakyReLU(0.2)
        )

    def construct(self, x):
        """EdgeConv construct."""
        get_graph_feature = Getgraphfeature(x, k=self.k)
        x = get_graph_feature.func()
        x = self.mlp(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x
