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
"""Video Swin Transformer."""

from typing import Optional
import ml_collections as collections

from mindspore import nn

from mindvision.classification.models.head import DenseHead
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel
from mindvision.msvideo.models.backbones import SwinTransformer3D
from mindvision.msvideo.models.base import BaseRecognizer
from mindvision.msvideo.models.embedding import PatchEmbed3D
from mindvision.msvideo.models.neck import GlobalAvgPooling3D

__all__ = [
    'swin3d',
    'swin3d_t',
    'swin3d_s',
    'swin3d_b',
    'swin3d_l',
]


def swin3d(num_classes: int,
           patch_size: int,
           window_size: int,
           embed_dim: int,
           depths: int,
           num_heads: int,
           representation_size: int,
           droppath_keep_prob: float,
           input_size: int = (32, 224, 224),
           in_channels: int = 3,
           mlp_ratio: float = 4.0,
           qkv_bias: bool = True,
           qk_scale: Optional[float] = None,
           keep_prob: float = 1.,
           attn_keep_prob: float = 1.,
           norm_layer: str = 'layer_norm',
           patch_norm: bool = True,
           pooling_keep_dim: bool = False,
           head_bias: bool = True,
           head_activation: Optional[str] = None,
           head_keep_prob: float = 0.5,
           pretrained: bool = False,
           arch: Optional[str] = None,
           ) -> nn.Cell:
    """
    Constructs a swin3d_tiny architecture from
    `Video Swin Transformer <http://arxiv.org/abs/2106.13230>`.

    Args:
        num_classes (int): The number of classification. Default: 400.
        patch_size (int): Patch size used by window attention. Default: (2, 4, 4).
        window_size (int): Window size used by window attention. Default: (8, 7, 7).
        embed_dim (int): Embedding dimension of the featrue generated from patch embedding layer. Default: 96.
        depths (int): Depths of each stage in Swin3d Tiny module. Default: (2, 2, 6, 2).
        num_heads (int): Numbers of heads of each stage in Swin3d Tiny module. Default: (3, 6, 12, 24).
        representation_size (int): Feature dimension of the last layer in backbone. Default: 768.
        droppath_keep_prob (float): The drop path keep probability. Default: 0.9.
        input_size (int | tuple(int)): Input feature size. Default: (32, 224, 224).
        in_channels (int): Input channels. Default: 3.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        qkv_bias (bool): If qkv_bias is True, add a learnable bias into query, key, value matrixes. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None.
        keep_prob (float): Dropout keep probability. Default: 1.0.
        attn_keep_prob (float): Keeping probability for attention dropout. Default: 1.0.
        norm_layer (string): Normalization layer. Default: 'layer_norm'.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        pooling_keep_dim (bool): Specifies whether to keep dimension shape the same as input feature. Default: False.
        head_bias (bool): Specifies whether the head uses a bias vector. Default: True.
        head_activation (Union[str, Cell, Primitive]): Activate function applied in the head. Default: None.
        head_keep_prob (float): Head's dropout keeping rate, between [0, 1]. Default: 0.5.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        arch (str | None, optional): Pre-trained model's architecture name. Default: None.
    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindvision.msvideo.models import swin3d
        >>>
        >>> net = swin3d(400, (2, 4, 4), (8, 7, 7), 96, (2, 2, 6, 2), (3, 6, 12, 24), 768, 0.9, False)
        >>> x = ms.Tensor(np.ones([1, 3, 32, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 400)

    About Swin:

    TODO: Swin3d introduction.

    Citation:

    .. code-block::

        @article{liu2021video,
            title={Video Swin Transformer},
            author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
            journal={arXiv preprint arXiv:2106.13230},
            year={2021}
        }
    """

    embedding = PatchEmbed3D(input_size=input_size,
                             patch_size=patch_size,
                             in_channels=in_channels,
                             embed_dim=embed_dim,
                             norm_layer=norm_layer
                             )
    backbone = SwinTransformer3D(input_size=embedding.output_size,
                                 embed_dim=embed_dim,
                                 depths=depths,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 keep_prob=keep_prob,
                                 attn_keep_prob=attn_keep_prob,
                                 droppath_keep_prob=droppath_keep_prob,
                                 norm_layer=norm_layer,
                                 patch_norm=patch_norm
                                 )
    neck = GlobalAvgPooling3D(keep_dims=pooling_keep_dim)
    head = DenseHead(input_channel=representation_size,
                     num_classes=num_classes,
                     has_bias=head_bias,
                     activation=head_activation,
                     keep_prob=head_keep_prob
                     )
    model = BaseRecognizer(backbone=backbone, embedding=embedding, neck=neck, head=head)
    if pretrained:
        # Download the pre-trained checkpoint file from url, and load ckpt file.
        # TODO: model_urls is not defined yet.
        LoadPretrainedModel(model, model_urls[arch]).run()
    return model


def swin3d_t(num_classes: int = 400,
             patch_size: int = (2, 4, 4),
             window_size: int = (8, 7, 7),
             embed_dim: int = 96,
             depths: int = (2, 2, 6, 2),
             num_heads: int = (3, 6, 12, 24),
             representation_size: int = 768,
             droppath_keep_prob: float = 0.9,
             pretrained: bool = False
             ) -> nn.Cell:
    """
    Video Swin Transformer Tiny (swin3d-T) model.
    """
    config = collections.ConfigDict()
    config.arch = "swin_tiny_" + "patch" + str(patch_size)
    config.num_classes = num_classes
    config.patch_size = patch_size
    config.window_size = window_size
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    config.representation_size = representation_size
    config.droppath_keep_prob = droppath_keep_prob
    config.pretrained = pretrained
    return swin3d(**config)


def swin3d_s(num_classes: int = 400,
             patch_size: int = (2, 4, 4),
             window_size: int = (8, 7, 7),
             embed_dim: int = 96,
             depths: int = (2, 2, 18, 2),
             num_heads: int = (3, 6, 12, 24),
             representation_size: int = 768,
             droppath_keep_prob: float = 0.9,
             pretrained: bool = False
             ) -> nn.Cell:
    """
    Video Swin Transformer Small (swin3d-S) model.
    """
    config = collections.ConfigDict()
    config.arch = "swin_tiny_" + "patch" + str(patch_size)
    config.num_classes = num_classes
    config.patch_size = patch_size
    config.window_size = window_size
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    config.representation_size = representation_size
    config.droppath_keep_prob = droppath_keep_prob
    config.pretrained = pretrained
    return swin3d(**config)


def swin3d_b(num_classes: int = 400,
             patch_size: int = (2, 4, 4),
             window_size: int = (8, 7, 7),
             embed_dim: int = 128,
             depths: int = (2, 2, 18, 2),
             num_heads: int = (4, 8, 16, 32),
             representation_size: int = 1024,
             droppath_keep_prob: float = 0.7,
             pretrained: bool = False
             ) -> nn.Cell:
    """
    Video Swin Transformer Base (swin3d-B) model.
    """
    config = collections.ConfigDict()
    config.arch = "swin_tiny_" + "patch" + str(patch_size)
    config.num_classes = num_classes
    config.patch_size = patch_size
    config.window_size = window_size
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    config.representation_size = representation_size
    config.droppath_keep_prob = droppath_keep_prob
    config.pretrained = pretrained
    return swin3d(**config)


def swin3d_l(num_classes: int = 400,
             patch_size: int = (2, 4, 4),
             window_size: int = (8, 7, 7),
             embed_dim: int = 192,
             depths: int = (2, 2, 18, 2),
             num_heads: int = (6, 12, 24, 48),
             representation_size: int = 1536,
             droppath_keep_prob: float = 0.9,
             pretrained: bool = False
             ) -> nn.Cell:
    """
    Video Swin Transformer Large (swin3d-L) model.
    """
    config = collections.ConfigDict()
    config.arch = "swin_tiny_" + "patch" + str(patch_size)
    config.num_classes = num_classes
    config.patch_size = patch_size
    config.window_size = window_size
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    config.representation_size = representation_size
    config.droppath_keep_prob = droppath_keep_prob
    config.pretrained = pretrained
    return swin3d(**config)
