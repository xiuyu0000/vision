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
"""VisTR embeding."""
import math
from typing import Optional
import mindspore
from mindspore import numpy as np
from mindspore import nn, ops
from mindspore.ops import operations as P
from mindvision.classification.models.backbones import resnet


class EmbedingBase(nn.Cell):
    r"""vistr embeding base.resnet+PositionEmbeddingSine,
       and perform other operations so that the output conforms to TransformerEncoder.
    Args:
        embeding(Optional[nn.Cell]):resnet50/101
        train_embeding(bool):Choose whether the network is in training
        num_channels(int):number of channel
        num_pos_feats(int):positional encoding length for each dimension.Default: 64
        num_frames(int):number of frame.Default: 36
        temperature(int):Adding coefficients makes the coding distribution more reasonable.Default: 10000
        normalize(bool):Choose whether to normalize.Default:True
        scale(float):Coefficients added for encoding.
        hidden_dim(int):Dimensions of the Transformer input vector.Default:384
        num_queries:number of object queries, ie detection slot. This is the maximal number of objects
                    VisTR can detect in a video. For ytvos, we recommend 10 queries for each frame,
                    thus 360 queries for 36 frames.Default: 360
    Returns:
        Tensor, output 4 tensors.
    """

    def __init__(self, embeding: Optional[nn.Cell], train_embeding: bool,
                 num_channels: int, num_pos_feats: int = 64,
                 num_frames: int = 36, temperature: int = 10000,
                 normalize: bool = True, scale: float = None,
                 hidden_dim: int = 384, num_queries: int = 360):
        super().__init__()
        for params in embeding.get_parameters():
            if (not train_embeding or 'layer2' not in params.name and
                    'layer3' not in params.name and
                    'layer4' not in params.name):
                params.requires_grad = False
            if 'beta' in params.name:
                params.requires_grad = False
            if 'gamma' in params.name:
                params.requires_grad = False
        self.body = embeding
        self.num_channels = num_channels
        self.cast = ops.Cast()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.frames = num_frames
        self.cumsum = ops.CumSum()
        self.reshape = ops.Reshape()
        self.stack = ops.Stack(axis=5)
        self.concat = ops.Concat(axis=4)
        self.transpose = ops.Transpose()
        self.sin = ops.Sin()
        self.cos = ops.Cos()
        dim_t = np.arange(0, self.num_pos_feats, dtype=mindspore.float32)
        self.dim_t = (self.cast(self.temperature, mindspore.float32) **
                      (2 * (dim_t // 2) / self.num_pos_feats))
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1,
                                    pad_mode='valid', has_bias=True)
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def construct(self, tensor, mask):
        """construct vistr embeding.
        """
        src_list = []
        pos_list = []
        features = []
        src = self.body.conv1(tensor)
        src = self.body.max_pool(src)
        src = self.body.layer1(src)
        src_list.append(src)
        src = self.body.layer2(src)
        src_list.append(src)
        src = self.body.layer3(src)
        src_list.append(src)
        src = self.body.layer4(src)
        src_list.append(src)
        for x in src_list:
            interpolate = P.ResizeNearestNeighbor(x.shape[-2:])
            ms = interpolate(mask[None])
            ms = self.cast(ms, mindspore.bool_)[0]
            features.append((x, ms))
            features_pos = self.PositionEmbeddingSine(ms)
            pos_list.append(features_pos)

        src, ms = features[-1]
        src_proj = self.input_proj(src)
        src_copy = src_proj.copy()
        n, c, h, w = src_proj.shape
        src_proj = src_proj.reshape(n//self.frames, self.frames, c, h, w)
        input_perm = (0, 2, 1, 3, 4)
        src_proj = self.transpose(src_proj, input_perm)
        src_proj = self.reshape(src_proj, (src_proj.shape[0],
                                           src_proj.shape[1],
                                           src_proj.shape[2],
                                           src_proj.shape[3]*src_proj.shape[4]))
        ms = self.reshape(ms, (n//self.frames, self.frames, h*w))
        pos = self.transpose(pos_list[-1], input_perm)
        pos = self.reshape(pos, (pos.shape[0], pos.shape[1], pos.shape[2],
                                 pos.shape[3]*pos.shape[4]))
        embedding_table = self.query_embed.embedding_table

        return src_proj, ms, embedding_table.copy(), pos, features, src_copy

    def PositionEmbeddingSine(self, mask):
        """Sine encoding
        """
        n, h, w = mask.shape
        mask = self.reshape(mask, (n//self.frames, self.frames, h, w))
        mask_copy = self.fill(mindspore.float32, mask.shape, 1)
        not_mask = mask_copy.masked_fill(mask, 0)
        z_embed = self.cumsum(not_mask, 1)
        y_embed = self.cumsum(not_mask, 2)
        x_embed = self.cumsum(not_mask, 3)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale
        pos_x = x_embed[:, :, :, :, None] / self.dim_t
        pos_y = y_embed[:, :, :, :, None] / self.dim_t
        pos_z = z_embed[:, :, :, :, None] / self.dim_t
        pos_x = self.stack([self.sin(pos_x[:, :, :, :, 0::2]),
                            self.cos(pos_x[:, :, :, :, 1::2])])
        pos_x = self.reshape(pos_x, (pos_x.shape[0], pos_x.shape[1], pos_x.shape[2],
                                     pos_x.shape[3], pos_x.shape[4]*pos_x.shape[5]))
        pos_y = self.stack([self.sin(pos_y[:, :, :, :, 0::2]),
                            self.cos(pos_y[:, :, :, :, 1::2])])
        pos_y = self.reshape(pos_y, (pos_y.shape[0], pos_y.shape[1], pos_y.shape[2],
                                     pos_y.shape[3], pos_y.shape[4]*pos_y.shape[5]))
        pos_z = self.stack([self.sin(pos_z[:, :, :, :, 0::2]),
                            self.cos(pos_z[:, :, :, :, 1::2])])
        pos_z = self.reshape(pos_z, (pos_z.shape[0], pos_z.shape[1], pos_z.shape[2],
                                     pos_z.shape[3], pos_z.shape[4]*pos_z.shape[5]))
        pos = self.concat((pos_z, pos_y, pos_x))
        input_perm = (0, 1, 4, 2, 3)
        pos = self.transpose(pos, input_perm)
        return pos


class VistrEmbeding(EmbedingBase):
    """vistr embed
    Args:
        name(str):name of resnet
    Returns:
        Tensor, output 3 tensors.
    Examples:
        >>> VisTR_emdeding('ResNet101', True)
    """

    def __init__(self, name: str, train_embeding: bool,
                 num_pos_feats: int = 64, num_frames: int = 36,
                 temperature: int = 10000, normalize: bool = True,
                 scale: float = None, hidden_dim: int = 384):
        embeding = resnet.ResNet101()
        if name == "ResNet101":
            embeding = resnet.ResNet101()
        if name == "ResNet50":
            embeding = resnet.ResNet50()
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        num_pos_feats = hidden_dim // 3
        super().__init__(embeding, train_embeding, num_channels, num_pos_feats,
                         num_frames, temperature, normalize, scale, hidden_dim)
