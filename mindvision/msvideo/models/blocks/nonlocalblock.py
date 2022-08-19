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
"""nonlocal block."""

import mindspore
from mindspore import nn
from mindspore import ops
from mindvision.msvideo.models.neck.pooling3d import MaxPooling3D


class NonLocalBlockND(nn.Cell):
    r"""
    Classification backbone for nonlocal.
    Implementation of Non-Local Block with 4 different pairwise functions.

    Applies Non-Local Block over 5D input (a mini-batch of 3D inputs with additional channel dimension).
    .. math::
        embedded_gaussian:
        f(x_i, x_j)=e^{\theta(x_i)^{T} \phi(x_j)}.
        gaussian:
        f(x_i, x_j)=e^{{x_i}^{T} {x_j}}.
        concatenation:
        f(x_i, x_j)=\{ReLU}({w_f}^{T}[\theta(x_i), \phi(x_j)]).
        dot_product:
        f(x_i, x_j)=\theta(x_i)^{T} \phi(x_j).

    Args:
        in_channels (int): original channel size.
        inter_channels (int): channel size inside the block if not specified reduced to half.
        mode: 4 mode to choose (gaussian, embedded, dot, and concatenation).
        sub_sample: whether to apply max pooling after pairwise.
        bn_layer: whether to add batch norm.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`.

    Examples:
        >>> net = nn.NonLocalBlockND(in_channels=3, bn_layer=bn_layer)
        >>> x = zeros((2, 3, 8, 20, 20), mindspore.float32)
        >>> output = net(x).shape
        >>> print(output)
        (2, 3, 8, 20, 20)
        """

    def __init__(
            self,
            in_channels,
            inter_channels=None,
            mode='embedded',
            sub_sample=True,
            bn_layer=True):

        super(NonLocalBlockND, self).__init__()

        mode_list = ['gaussian', 'embedded', 'dot', 'concatenation']
        check_string(mode, mode_list)

        self.mode = mode
        self.transpose = ops.Transpose()
        self.batmatmul = ops.BatchMatMul()
        self.tile = ops.Tile()
        self.concat_op = ops.Concat(1)
        self.zeros = ops.Zeros()
        self.softmax = ops.Softmax(axis=-1)

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv3d(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           has_bias=True
                           )

        if bn_layer:
            self.w = nn.SequentialCell(
                nn.Conv3d(in_channels=self.inter_channels,
                          out_channels=self.in_channels,
                          kernel_size=1),
                nn.BatchNorm3d(self.in_channels))
        else:
            self.w = nn.Conv3d(in_channels=self.inter_channels,
                               out_channels=self.in_channels,
                               kernel_size=1,
                               has_bias=True)
        if self.mode in ["embedded", "dot", "concatenation"]:
            self.theta = nn.Conv3d(in_channels=self.in_channels,
                                   out_channels=self.inter_channels,
                                   kernel_size=1,
                                   has_bias=True)
            self.phi = nn.Conv3d(in_channels=self.in_channels,
                                 out_channels=self.inter_channels,
                                 kernel_size=1,
                                 has_bias=True)
        if self.mode == "concatenation":
            self.concat_project = nn.SequentialCell(
                nn.Conv2d(
                    self.inter_channels * 2,
                    out_channels=1,
                    kernel_size=1,
                    pad_mode='same',
                    has_bias=False),
                nn.ReLU()
            )

        if sub_sample:
            max_pool_layer = MaxPooling3D(kernel_size=(1, 2, 2), strides=(1, 2, 2))
            self.g = nn.SequentialCell(self.g, max_pool_layer)
            if self.mode != 'guassian':
                self.phi = nn.SequentialCell(self.phi, max_pool_layer)
            else:
                self.phi = self.phi

    def construct(self, x):
        """non-local block construct."""
        batch_size = x.shape[0]
        g_x = self.g(x).view((batch_size, self.inter_channels, -1))
        input_perm = (0, 2, 1)
        g_x = self.transpose(g_x, input_perm)
        f = self.zeros((1, 1, 1), mindspore.float32)

        if self.mode == "gaussian":
            theta_x = x.view((batch_size, self.in_channels, -1))
            theta_x = self.transpose(theta_x, input_perm)
            phi_x = x.view(batch_size, self.in_channels, -1)
            f = self.batmatmul(theta_x, phi_x)
        elif self.mode in ["embedded", "dot"]:
            theta_x = self.theta(x).view((batch_size, self.inter_channels, -1))
            theta_x = self.transpose(theta_x, input_perm)
            phi_x = self.phi(x).view((batch_size, self.inter_channels, -1))
            f = self.batmatmul(theta_x, phi_x)
        elif self.mode == "concatenation":
            theta_x = self.theta(x).view(
                (batch_size, self.inter_channels, -1, 1))
            phi_x = self.phi(x).view((batch_size, self.inter_channels, 1, -1))
            h = theta_x.shape[2]
            w = phi_x.shape[3]
            theta_x = self.tile(theta_x, (1, 1, 1, w))
            phi_x = self.tile(phi_x, (1, 1, h, 1))
            concat_feature = self.concat_op((theta_x, phi_x))
            f = self.concat_project(concat_feature)
            b, _, h, w = f.shape
            f = f.view((b, h, w))
        f_div_c = self.zeros((1, 1, 1), mindspore.float32)
        if self.mode in ["gaussian", "embedded"]:
            f_div_c = self.softmax(f)
        elif self.mode in ["dot", "concatenation"]:
            n = f.shape[-1]
            f_div_c = f / n

        y = self.batmatmul(f_div_c, g_x)
        y = self.transpose(y, input_perm)
        y = y.view((batch_size,
                    self.inter_channels,
                    x.shape[2],
                    x.shape[3],
                    x.shape[4]))
        w_y = self.w(y)
        z = w_y + x
        return z
