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
"""Swin Transformer 3D block."""

from typing import Optional
from mindspore import nn
from mindspore import ops

from mindvision.check_param import Rel
from mindvision.check_param import Validator
from mindvision.classification.models.blocks.feed_forward import FeedForward
from mindvision.msvideo.engine.ops import Roll3D
from mindvision.msvideo.models.blocks import Identity
from mindvision.msvideo.models.blocks import ProbDropPath3D
from mindvision.msvideo.models.blocks import WindowAttention3D
from mindvision.msvideo.utils import compute_mask
from mindvision.msvideo.utils import limit_window_size
from mindvision.msvideo.utils import window_partition
from mindvision.msvideo.utils import window_reverse


class SwinTransformerBlock3D(nn.Cell):
    """
    A Video Swin Transformer Block. The implementation of this block follows
    the paper "Video Swin Transformer".

    Args:
        embed_dim (int): input feature's embedding dimension, namely, channel number. Default: 96.
        input_size (int | tuple(int)): input feature size. Default: (16, 56, 56).
        num_head (int): number of attention head of the current Swin3d block. Default: 3.
        window_size (int): window size of window attention. Default: (8, 7, 7).
        shift_size (tuple[int]): shift size for shifted window attention. Default: (4, 3, 3).
        mlp_ratio (float): ratio of mlp hidden dim to embedding dim. Default: 4.0.
        qkv_bias (bool): if True, add a learnable bias to query, key,value. Default: True.
        qk_scale (float | None, optional): override default qk scale of head_dim ** -0.5 if set True. Default: None.
        keep_prob (float): dropout keep probability. Default: 1.0.
        attn_keep_prob (float): units keeping probability for attention dropout. Default: 1.0.
        droppath_keep_prob (float): path keeping probability for stochastic droppath. Default: 1.0.
        act_layer (nn.Cell): activation layer. Default: nn.GELU.
        norm_layer (nn.Cell): normalization layer. Default: 'layer_norm'.

    Inputs:
        - **x** (Tensor) - Input feature of shape (B, D, H, W, C).
        - **mask_matrix** (Tensor) - Attention mask for cyclic shift.

    Outputs:
        Tensor of shape (B, D, H, W, C)

    Examples:
        >>> net1 = SwinTransformerBlock3D()
        >>> input = ops.Zeros()((8,16,56,56,96), mindspore.float32)
        >>> output = net1(input, None)
        >>> print(output.shape)
    """

    def __init__(self,
                 embed_dim: int = 96,
                 input_size: int = (16, 56, 56),
                 num_head: int = 3,
                 window_size: int = (8, 7, 7),
                 shift_size: int = (4, 3, 3),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 keep_prob: float = 1.,
                 attn_keep_prob: float = 1.,
                 droppath_keep_prob: float = 1.,
                 act_layer: nn.Cell = nn.GELU,
                 norm_layer: str = 'layer_norm'
                 ):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        # get window size and shift size
        self.window_size, self.shift_size = limit_window_size(
            input_size, window_size, shift_size)
        # check self.shift_size whether is smaller than self.window_size and
        # larger than 0
        Validator.check_int_range(
            self.shift_size[0], 0, self.window_size[0],
            Rel.INC_LEFT,
            arg_name="shift size", prim_name="SwinTransformerBlock3D")
        Validator.check_int_range(
            self.shift_size[1], 0, self.window_size[1],
            Rel.INC_LEFT, arg_name="shift size",
            prim_name="SwinTransformerBlock3D")
        Validator.check_int_range(
            self.shift_size[2], 0, self.window_size[2],
            Rel.INC_LEFT, arg_name="shift size",
            prim_name="SwinTransformerBlock3D")

        self.embed_dim = embed_dim
        self.num_head = num_head
        self.mlp_ratio = mlp_ratio
        if isinstance(self.embed_dim, int):
            self.embed_dim = (self.embed_dim,)

        # the first layer norm
        if norm_layer == 'layer_norm':
            self.norm1 = nn.LayerNorm(self.embed_dim, epsilon=1e-5)
        else:
            self.norm1 = Identity()
        # the second layer norm
        if norm_layer == 'layer_norm':
            self.norm2 = nn.LayerNorm(self.embed_dim, epsilon=1e-5)
        else:
            self.norm2 = Identity()

        # window attention 3D block
        self.attn = WindowAttention3D(self.embed_dim[0],
                                      window_size=self.window_size,
                                      num_head=num_head,
                                      qkv_bias=qkv_bias,
                                      qk_scale=qk_scale,
                                      attn_kepp_prob=attn_keep_prob,
                                      proj_keep_prob=keep_prob
                                      )

        self.drop_path = ProbDropPath3D(
            droppath_keep_prob) if droppath_keep_prob < 1. else Identity()

        mlp_hidden_dim = int(self.embed_dim[0] * mlp_ratio)

        # reuse classification.models.block.feed_forward as MLP here
        self.mlp = FeedForward(in_features=self.embed_dim[0],
                               hidden_features=mlp_hidden_dim,
                               activation=act_layer,
                               keep_prob=keep_prob
                               )

    def _construc_part1(self, x, mask_matrix):
        """"Construct W-MSA and SW-MSA."""
        batch_size, depth, height, width, channel_num = x.shape
        window_size = self.window_size
        shift_size = self.shift_size
        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - depth % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - height % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - width % window_size[2]) % window_size[2]
        x_padded = []
        pad = nn.Pad(paddings=(
            (pad_d0, pad_d1),
            (pad_t, pad_b),
            (pad_l, pad_r),
            (0, 0)))
        for i in range(x.shape[0]):
            x_b = x[i]
            x_b = pad(x_b)
            x_padded.append(x_b)
        x = ops.Stack(axis=0)(x_padded)
        # cyclic shift
        _, t_padded, h_padded, w_padded, _ = x.shape
        if [i for i in shift_size if i > 0]:
            shifted_x = Roll3D(shift=[-i for i in shift_size])(x)
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows: (B*nW, Wd*Wh*Ww, C)
        x_windows = window_partition(shifted_x, window_size)
        # W-MSA/SW-MSA
        # attn_windows: (B*nW, Wd*Wh*Ww, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (channel_num,)))
        shifted_x = window_reverse(attn_windows, window_size, batch_size,
                                   t_padded, h_padded, w_padded)
        # reverse cyclic shift
        if [i for i in shift_size if i > 0]:
            x = Roll3D(shift=shift_size)(shifted_x)
        else:
            x = shifted_x
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :depth, :height, :width, :]
        return x

    def _construct_part2(self, x):
        """Construct MLP."""
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        return x

    def construct(self, x, mask_matrix=None):
        """Construct 3D Swin Transformer Block."""
        shortcut = x
        x = self._construc_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        x = x + self._construct_part2(x)
        return x


class PatchMerging(nn.Cell):
    """
    Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Cell): Normalization layer. Default: nn.LayerNorm。

    Inputs:
        - **x** (Tensor) - Input feature of shape (B, D, H, W, C).

    Outputs:
        Tensor of shape (B, D, H/2, W/2, 2*C)
    """

    def __init__(self,
                 dim: int = 96,
                 norm_layer: str = 'layer_norm'):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Dense(4 * self.dim, 2 * self.dim, has_bias=False)
        if norm_layer == 'layer_norm':
            self.norm = nn.LayerNorm((4 * self.dim,))
        else:
            self.norm = Identity()

    def construct(self, x):
        """Construct Patch Merging Layer."""

        batch_size, _, height, width, _ = x.shape

        # padding
        pad_input = (height % 2 == 1) or (width % 2 == 1)
        if pad_input:
            x_padded = []
            pad = nn.Pad(
                paddings=(
                    (0, 0), (0, height %
                             2), (0, width %
                                  2), (0, 0)))
            for i in range(batch_size):
                x_b = x[i]
                x_b = pad(x_b)
                x_padded.append(x_b)
            x_padded = ops.Stack(axis=0)(x_padded)
            x = x_padded
        # x_0, x_1, x_2, x_3: (B, D, H/2, W/2, C)
        x_0 = x[:, :, 0::2, 0::2, :]
        x_1 = x[:, :, 1::2, 0::2, :]
        x_2 = x[:, :, 0::2, 1::2, :]
        x_3 = x[:, :, 1::2, 1::2, :]
        # x: (B, D, H/2, W/2, 4*C)
        x = ops.Concat(axis=-1)([x_0, x_1, x_2, x_3])

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SwinTransformerStage3D(nn.Cell):
    r"""
    A basic Swin Transformer layer for one stage.

    Args:
        embed_dim (int): input feature's embedding dimension, namely, channel number. Default: 96.
        input_size (tuple[int]): input feature size. Default. (16, 56, 56).
        depth (int): depth of the current Swin3d stage. Default: 2.
        num_head (int): number of attention head of the current Swin3d stage. Default: 3.
        window_size (int): window size of window attention. Default: (8, 7, 7).
        mlp_ratio (float): ratio of mlp hidden dim to embedding dim. Default: 4.0.
        qkv_bias (bool): if qkv_bias is True, add a learnable bias into query, key, value matrixes. Default: Truee
        qk_scale (float | None, optional): override default qk scale of head_dim ** -0.5 if set. Default: None.
        keep_prob (float): dropout keep probability. Default: 1.0.
        attn_keep_prob (float): units keeping probability for attention dropout. Default: 1.
        droppath_keep_prob (float): path keeping probability for stochastic droppath. Default: 0.8.
        norm_layer(string): normalization layer. Default: 'layer_norm'.
        downsample (nn.Cell | None, optional): downsample layer at the end of swin3d stage. Default: PatchMerging.

    Inputs:
        A video feature of shape (N, D, H, W, C)
    Returns:
        Tensor of shape (N, D, H / 2, W / 2, 2 * C)
    """

    def __init__(self,
                 embed_dim=96,
                 input_size=(16, 56, 56),
                 depth=2,
                 num_head=3,
                 window_size=(8, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 keep_prob=1.,
                 attn_keep_prob=1.,
                 droppath_keep_prob=0.8,
                 norm_layer='layer_norm',
                 downsample=PatchMerging
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        # build blocks
        self.blocks = nn.CellList([
            SwinTransformerBlock3D(
                embed_dim=embed_dim,
                num_head=num_head,
                input_size=input_size,
                window_size=self.window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                keep_prob=keep_prob,
                attn_keep_prob=attn_keep_prob,
                droppath_keep_prob=droppath_keep_prob[i] if isinstance(
                    droppath_keep_prob, list) else droppath_keep_prob,
                norm_layer=norm_layer
            )
            for i in range(depth)])
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=embed_dim, norm_layer=norm_layer)
        self.window_size, self.shift_size = limit_window_size(
            input_size, self.window_size, self.shift_size)
        self.attn_mask = compute_mask(
            input_size[0], input_size[1], input_size[2],
            self.window_size, self.shift_size)

    def construct(self, x):
        """Construct a basic stage layer for VideoSwinTransformer."""
        for blk in self.blocks:
            x = blk(x, self.attn_mask)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
