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
"""Swin Transformer 3D block."""

from mindspore import nn
from mindspore import ops

from mindvision.msvideo.models.blocks import Identity


class PatchEmbed3D(nn.Cell):
    """
    Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_channels (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.

    Inputs:
        An original Video tensor in data format of 'NCDHW'.

    Returns:
        An embedded tensor in data format of 'NDHWC'.
    """

    def __init__(self, input_size=(16, 224, 224), patch_size=(2, 4, 4),
                 in_channels=3, embed_dim=96, norm_layer='layer_norm'):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.project = nn.Conv3d(in_channels, embed_dim, has_bias=True,
                                 kernel_size=patch_size, stride=patch_size)

        if norm_layer == 'layer_norm':
            if isinstance(self.embed_dim, int):
                self.embed_dim = (self.embed_dim,)
            self.norm = nn.LayerNorm(self.embed_dim)
        else:
            self.norm = Identity()

        self.output_size = [input_size[0] // patch_size[0],
                            input_size[1] // patch_size[1],
                            input_size[2] // patch_size[2]]

    def construct(self, x):
        """Construct Patch Embedding for 3D features."""
        # padding
        _, _, depth, height, width = x.shape
        pad_d = 0
        pad_b = 0
        pad_r = 0
        x_padded = []
        if width % self.patch_size[2] != 0:
            pad_r = self.patch_size[2] - width % self.patch_size[2]
        if height % self.patch_size[1] != 0:
            pad_b = self.patch_size[1] - height % self.patch_size[1]
        if depth % self.patch_size[0] != 0:
            pad_d = self.patch_size[0] - depth % self.patch_size[0]
        pad = nn.Pad(paddings=(
            (0, pad_d),
            (0, pad_b),
            (0, pad_r),
            (0, 0)
        ))
        for i in range(x.shape[0]):
            x_b = x[i]
            x_b = pad(x_b)
            x_padded.append(x_b)
        x = ops.Stack(axis=0)(x_padded)
        x = self.project(x)  # B C D Wh Ww
        batch_size, channel_num, depth_w, height_w, width_w = x.shape
        x = x.reshape(batch_size, channel_num, -1).transpose(0, 2, 1)
        x = self.norm(x)
        x = x.view(-1, depth_w, height_w, width_w, channel_num)
        return x
