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
"""Swin Transformer 3D Backbone."""

import numpy as np
from mindspore import nn

from mindvision.engine.class_factory import ClassFactory
from mindvision.engine.class_factory import ModuleType
from mindvision.msvideo.models.blocks import Identity
from mindvision.msvideo.models.blocks import PatchMerging
from mindvision.msvideo.models.blocks import SwinTransformerStage3D


@ClassFactory.register(ModuleType.BACKBONE)
class SwinTransformer3D(nn.Cell):
    """
    Video Swin Transformer backbone.
    A mindspore implementation of : `Video Swin Transformer` http://arxiv.org/abs/2106.13230

    TODO: Code comments should correspond to input parameters. such as: input_size.
    Args:
        input_size (int | tuple(int)): input feature size. Default: (16, 56, 56).
        embed_dim (int): input feature's embedding dimension, namely, channel number. Default: 96.
        depths (tuple[int]): depths of each Swin3d stage. Default: (2, 2, 6, 2).
        num_heads (tuple[int]): number of attention head of each Swin3d stage. Default: (3, 6, 12, 24).
        window_size (int): window size of window attention. Default: (8, 7, 7).
        mlp_ratio (float): ratio of mlp hidden dim to embedding dim. Default: 4.0.
        qkv_bias (bool): if qkv_bias is True, add a learnable bias into query, key, value matrixes. Default: True.
        qk_scale (float | None, optional): override default qk scale of head_dim ** -0.5 if set. Default: None.
        keep_prob (float): dropout keep probability. Default: 1.0.
        attn_keep_prob (float): units keeping probability for attention dropout. Default: 1.
        droppath_keep_prob (float): path keeping probability for stochastic droppath. Default: 0.8.
        norm_layer (string): normalization layer. Default: 'layer_norm'.
        patch_norm (bool): if True, add normalization after patch embedding. Default: True.
    """

    def __init__(self,
                 input_size=(16, 56, 56),
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=(8, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 keep_prob=1.,
                 attn_keep_prob=1.,
                 droppath_keep_prob=0.8,
                 norm_layer='layer_norm',
                 patch_norm=True
                 ):
        super(SwinTransformer3D, self).__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.input_size = input_size
        self.pos_drop = nn.Dropout(keep_prob=keep_prob)

        # stochastic depth decay rule
        dpr = list(np.linspace(1, droppath_keep_prob, sum(depths)))

        # build layers
        self.layers = nn.CellList()
        for i_layer in range(self.num_layers):
            layer = SwinTransformerStage3D(
                embed_dim=int(embed_dim * 2 ** i_layer),
                input_size=(self.input_size[0],
                            self.input_size[1] // (2 ** i_layer),
                            self.input_size[2] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_head=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                keep_prob=keep_prob,
                attn_keep_prob=attn_keep_prob,
                droppath_keep_prob=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1
                else None
            )
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # add a norm layer for each output
        if norm_layer == 'layer_norm':
            self.norm = nn.LayerNorm((self.num_features,))
        else:
            self.norm = Identity()

    def construct(self, x):
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.transpose(0, 4, 1, 2, 3)
        return x
