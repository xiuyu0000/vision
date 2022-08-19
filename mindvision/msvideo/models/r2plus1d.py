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
""" nonlocal network."""

from typing import List, Optional, Tuple

from mindspore import nn
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.utils.model_urls import model_urls
from mindvision.msvideo.models.backbones import R2Plus1dNet
from mindvision.classification.models.head import DenseHead
from mindvision.msvideo.models.backbones.resnet3d import ResidualBlockBase3D, ResidualBlock3D
from mindvision.utils.load_pretrained_model import LoadPretrainedModel
from mindvision.msvideo.models.neck.poolflatten import AvgpoolFlatten

__all__ = ['r2plus1d',
           'r2plus1d18',
           'r2plus1d50']


def r2plus1d(block: Optional[nn.Cell],
             layer_nums: List[int],
             stage_channels: Tuple[int],
             stage_strides: Tuple[Tuple[int]],
             num_classes: int,
             keep_prob: float
             ) -> R2Plus1dNet:
    """R(2+1)D model.

    Du Tran.
    "A Closer Look at Spatiotemporal Convolutions for Action Recognition."
    https://arxiv.org/abs/1711.11248

    Args:
        block (Optional[nn.Cell]): THe block for network.
        layer_nums (list[int]): The numbers of block in different layers.
        stage_channels (Tuple[int]): Output channel for every res stage..
        stage_strides (Tuple[Tuple[int]]): Strides for every res stage.
        num_classes (int): The channel dimensions of the output.
        keep_prob (float): Dropout rate for fc layer. If equal to 0.0, perform no dropout.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> from mindvision.msvideo.models import r2plus1d
        >>> from mindvision.msvideo.models.backbones.resnet3d import ResidualBlockBase3D
        >>> data = Tensor(np.random.randn(2, 3, 16, 112, 112), dtype=mindspore.float32)
        >>>
        >>> net = r2plus1d(block=ResidualBlockBase3D,
                           stage_channels=(64, 128, 256, 512),
                           stage_strides=((1, 1, 1),
                                          (2, 2, 2),
                                          (2, 2, 2),
                                          (2, 2, 2)),
                           num_classes: int = 400,
                           keep_prob: float = 0.5)
        >>>
        >>> predict = net(data)
        >>> print(predict.shape)

    About R(2+1)D:

    TODO: R(2+1)D introduction.

    Citation:

    .. code-block::

        @inproceedings{tran2018closer,
            Title    = {A closer look at spatiotemporal convolutions for action recognition},
            Author   = {Tran, Du and Wang, Heng and Torresani, Lorenzo and Ray, Jamie and LeCun, Yann and
                        Paluri, Manohar},
            Booktitle= {Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
            Pages    = {6450--6459},
            Year     = {2018}
        }
    """

    backbone = R2Plus1dNet(block=block,
                           layer_nums=layer_nums,
                           stage_channels=stage_channels,
                           stage_strides=stage_strides)
    neck = AvgpoolFlatten()
    head = DenseHead(input_channel=stage_channels[-1]*block.expansion,
                     num_classes=num_classes,
                     keep_prob=keep_prob)
    model = BaseClassifier(backbone, neck, head)

    return model


def r2plus1d18(block: Optional[nn.Cell] = ResidualBlockBase3D,
               stage_channels: Tuple[int] = (64, 128, 256, 512),
               stage_strides: Tuple[Tuple[int]] = ((1, 1, 1),
                                                   (2, 2, 2),
                                                   (2, 2, 2),
                                                   (2, 2, 2)),
               num_classes: int = 400,
               keep_prob: float = 0.5,
               pretrained: bool = False):
    """R(2+1)D-18 model"""

    model = r2plus1d(block, [2, 2, 2, 2], stage_channels, stage_strides, num_classes, keep_prob)
    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        arch = "r2plus1d18_kinetics400"
        LoadPretrainedModel(model, model_urls[arch]).run()
    return model


def r2plus1d50(block: Optional[nn.Cell] = ResidualBlock3D,
               stage_channels: Tuple[int] = (64, 128, 256, 512),
               stage_strides: Tuple[Tuple[int]] = ((1, 1, 1),
                                                   (2, 2, 2),
                                                   (2, 2, 2),
                                                   (2, 2, 2)),
               num_classes: int = 400,
               keep_prob: float = 0.5,
               pretrained: bool = False):
    """R(2+1)D-50 model"""

    model = r2plus1d(block, [3, 4, 6, 3], stage_channels, stage_strides, num_classes, keep_prob)
    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        arch = "r2plus1d50_kinetics400"
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model
