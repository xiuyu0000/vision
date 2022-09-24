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
""" X3D network."""

import math
from typing import Type

from mindvision.classification.models.classifiers.base import BaseClassifier
from mindvision.classification.models.head.multilayer_dense_head import MultilayerDenseHead
from mindvision.msvideo.models.backbones.x3d import ResNetX3D, BlockX3D
from mindvision.msvideo.models.neck.poolflatten import AvgpoolFlatten

__all__ = [
    'x3d_xs',
    'x3d_s',
    'x3d_m',
    'x3d_l'
]


def x3d(block: Type[BlockX3D],
        depth_factor: float,
        num_frames: int,
        train_crop_size: int,
        num_classes: int,
        dropout_rate: float,
        bottleneck_factor: float = 2.25
        ) -> ResNetX3D:
    """
    x3d architecture.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """
    block_basis = [1, 2, 5, 3]
    stage_channels = (24, 48, 96, 192)
    stage_strides = ((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2))
    drop_rates = [0.2, 0.3, 0.4, 0.5]
    layer_nums = []
    for item in block_basis:
        nums = int(math.ceil(item * depth_factor))
        layer_nums.append(nums)
    spat_sz = int(math.ceil(train_crop_size / 32.0))
    pool_size = [num_frames, spat_sz, spat_sz]
    input_channel = int(math.ceil(192 * bottleneck_factor))

    backbone = ResNetX3D(block=block, layer_nums=layer_nums, stage_channels=stage_channels,
                         stage_strides=stage_strides, drop_rates=drop_rates)
    neck = AvgpoolFlatten(pool_size)
    head = MultilayerDenseHead(input_channel, num_classes, [2048],
                               [1.0, dropout_rate], ['relu', None])

    model = BaseClassifier(backbone, neck, head)

    return model


def x3d_m(num_classes: int = 400,
          dropout_rate: float = 0.5,
          depth_factor: float = 2.2,
          num_frames: int = 16,
          train_crop_size: int = 224
          ) -> ResNetX3D:
    """
    X3D middle model.

    Christoph Feichtenhofer. "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730

    Args:
        num_classes (int): the channel dimensions of the output.
        dropout_rate (float): dropout rate. If equal to 0.0, perform no
            dropout.
        depth_factor (float): Depth expansion factor.
        num_frames (int): The number of frames of the input clip.
        train_crop_size (int): The spatial crop size for training.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindvision.msvideo.models import x3d_m
        >>>
        >>> network = x3d_m()
        >>> input_x = Tensor(np.random.randn(1, 3, 16, 224, 224).astype(np.float32))
        >>> out = network(input_x)
        >>> print(out.shape)
        (1, 400)

    About x3d: Expanding Architectures for Efficient Video Recognition.

    .. code-block::

        @inproceedings{x3d2020,
            Author    = {Christoph Feichtenhofer},
            Title     = {{X3D}: Progressive Network Expansion for Efficient Video Recognition},
            Booktitle = {{CVPR}},
            Year      = {2020}
        }
    """
    return x3d(BlockX3D, depth_factor, num_frames, train_crop_size, num_classes, dropout_rate)


def x3d_s(num_classes: int = 400,
          dropout_rate: float = 0.5,
          depth_factor: float = 2.2,
          num_frames: int = 13,
          train_crop_size: int = 160
          ) -> ResNetX3D:
    """
    X3D small model.
    """
    return x3d(BlockX3D, depth_factor, num_frames, train_crop_size, num_classes, dropout_rate)


def x3d_xs(num_classes: int = 400,
           dropout_rate: float = 0.5,
           depth_factor: float = 2.2,
           num_frames: int = 4,
           train_crop_size: int = 160
           ) -> ResNetX3D:
    """
    X3D x-small model.
    """
    return x3d(BlockX3D, depth_factor, num_frames, train_crop_size, num_classes, dropout_rate)


def x3d_l(num_classes: int = 400,
          dropout_rate: float = 0.5,
          depth_factor: float = 5.0,
          num_frames: int = 16,
          train_crop_size: int = 312
          ) -> ResNetX3D:
    """
    X3D large model.
    """
    return x3d(BlockX3D, depth_factor, num_frames, train_crop_size, num_classes, dropout_rate)
