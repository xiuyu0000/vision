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
"""Window attention block."""

from typing import Optional

import numpy as np

from mindspore import dtype
from mindspore import nn
from mindspore import ops
from mindspore import Parameter
from mindspore import Tensor


class WindowAttention3D(nn.Cell):
    r"""
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        in_channels (int): Number of input channels.
        window_size (tuple[int]): The depth length, height and width of the window. Default: (8, 7, 7).
        num_head (int): Number of attention heads. Default: 3.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None.
        attn_keep_prob (float, optional): Dropout keep ratio of attention weight. Default: 1.0.
        proj_keep_prob (float, optional): Dropout keep ratio of output. Deault: 1.0.

    Inputs:
        - `x` (Tensor) - Tensor of shape (B, N, C).
        - `mask` (Tensor) - (0 / - inf) mask with shape of (num_windows, N, N) or None.

    Outputs:
        Tensor of shape (B, N, C), which is equal to the input **x**.

    Examples:
        >>> input = ops.Zeros()((1024, 392, 96), mindspore.float32)
        >>> net = WindowAttention3D(96, (8, 7, 7), 3, True, None, 0., 0.)
        >>> output = net(input)
        >>> print(output)
        (1024, 392, 96)
    """

    def __init__(self,
                 in_channels: int = 96,
                 window_size: int = (8, 7, 7),
                 num_head: int = 3,
                 qkv_bias: Optional[bool] = True,
                 qk_scale: Optional[float] = None,
                 attn_kepp_prob: Optional[float] = 1.0,
                 proj_keep_prob: Optional[float] = 1.0
                 ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_head = num_head
        head_dim = in_channels // num_head
        self.scale = qk_scale or head_dim ** -0.5
        # define a parameter table of relative position bias
        init_tensor = np.random.randn(
            (2 * self.window_size[0] - 1)
            * (2 * self.window_size[1] - 1)
            * (2 * self.window_size[2] - 1),
            num_head
        )
        init_tensor = Tensor(init_tensor, dtype=dtype.float32)
        # self.relative_position_bias_table: [2*Wt-1 * 2*Wh-1 * 2*Ww-1, nH]
        self.relative_position_bias_table = Parameter(
            init_tensor
        )
        # get pair-wise relative position index for each token in a window
        coords_d = np.arange(self.window_size[0])
        coords_h = np.arange(self.window_size[1])
        coords_w = np.arange(self.window_size[2])
        # coords: [3, Wd, Wh, Ww]
        coords = np.stack(
            np.meshgrid(
                coords_d,
                coords_h,
                coords_w,
                indexing='ij'
            )
        )
        coords_flatten = np.reshape(coords, (coords.shape[0], -1))
        # relative_coords: [3, Wd*Wh*Ww, Wd*Wh*Ww]
        relative_coords = coords_flatten[:, :, np.newaxis] - \
            coords_flatten[:, np.newaxis, :]
        # relative_coords: [Wh*Ww, Wh*Ww, 2]
        relative_coords = relative_coords.transpose(1, 2, 0)
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * \
            (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        # self.relative_position_index: [Wd*Wh*Ww, Wd*Wh*Ww]
        self.relative_position_index = Parameter(Tensor(relative_coords.sum(-1)), requires_grad=False)
        # QKV Linear layer
        self.qkv = nn.Dense(in_channels, in_channels * 3, has_bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_kepp_prob)
        self.proj = nn.Dense(in_channels, in_channels)
        self.proj_dropout = nn.Dropout(proj_keep_prob)
        self.softmax = nn.Softmax(axis=-1)
        # ops definition
        self.batch_matmul = ops.BatchMatMul()
        self.expand_dims = ops.ExpandDims()
        self.reshape = ops.Reshape()

    def construct(self, x, mask=None):
        """Construct WindowAttention3D."""

        batch_size, window_num, channel_num = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, window_num, 3,
                          self.num_head, channel_num // self.num_head)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        # q, k, v: [B, nH, N, C]
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = query * self.scale
        attn = self.batch_matmul(query, key.transpose(0, 1, 3, 2))
        # relative_position_bias: [Wd*Wh*Ww, Wd*Wh*Ww, nH]
        relative_position_bias = self.relative_position_bias_table[
            self.reshape(
                self.relative_position_index[:window_num, :window_num], (-1,)
            )
        ]
        relative_position_bias = self.reshape(
            relative_position_bias, (window_num, window_num, -1)
        )
        relative_position_bias = relative_position_bias.transpose(2, 0, 1)
        relative_position_bias = self.expand_dims(relative_position_bias, 0)
        # attn: [B, nH, N, N]
        attn = attn + relative_position_bias
        # masked attention
        if mask is not None:
            n_w = mask.shape[0]
            mask = self.expand_dims(mask, 1)
            mask = self.expand_dims(mask, 0)
            attn = attn.view(batch_size // n_w, n_w, self.num_head,
                             window_num, window_num) + mask
            attn = attn.view(-1, self.num_head, window_num, window_num)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_dropout(attn)
        x = self.batch_matmul(attn, value).transpose(
            0, 2, 1, 3).reshape(batch_size, window_num, channel_num)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x
