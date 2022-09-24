# Copyright 2021
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
"""Mobilenet_v3."""

from typing import Optional

from mindspore import nn

from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.models.backbones import MobileNetV3
from mindvision.classification.models.head import MultilayerDenseHead
from mindvision.classification.models.neck import GlobalAvgPooling
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel

__all__ = ['mobilenet_v3_small',
           'mobilenet_v3_large']

model_cfgs = {
    "large": [
        [3, 16, 16, False, 'relu', 1],
        [3, 64, 24, False, 'relu', 2],
        [3, 72, 24, False, 'relu', 1],
        [5, 72, 40, True, 'relu', 2],
        [5, 120, 40, True, 'relu', 1],
        [5, 120, 40, True, 'relu', 1],
        [3, 240, 80, False, 'hswish', 2],
        [3, 200, 80, False, 'hswish', 1],
        [3, 184, 80, False, 'hswish', 1],
        [3, 184, 80, False, 'hswish', 1],
        [3, 480, 112, True, 'hswish', 1],
        [3, 672, 112, True, 'hswish', 1],
        [5, 672, 160, True, 'hswish', 2],
        [5, 960, 160, True, 'hswish', 1],
        [5, 960, 160, True, 'hswish', 1]
    ],
    "small": [
        [3, 16, 16, True, 'relu', 2],
        [3, 72, 24, False, 'relu', 2],
        [3, 88, 24, False, 'relu', 1],
        [5, 96, 40, True, 'hswish', 2],
        [5, 240, 40, True, 'hswish', 1],
        [5, 240, 40, True, 'hswish', 1],
        [5, 120, 48, True, 'hswish', 1],
        [5, 144, 48, True, 'hswish', 1],
        [5, 288, 96, True, 'hswish', 2],
        [5, 576, 96, True, 'hswish', 1],
        [5, 576, 96, True, 'hswish', 1]]
}


def mobilenet_v3_small(num_classes: int = 1000,
                       pretrained: bool = False,
                       multiplier: float = 1.0,
                       norm: Optional[nn.Cell] = None,
                       round_nearest: int = 8):
    """Mobilenet_v3 structure."""
    backbone = MobileNetV3(model_cfgs=model_cfgs['small'],
                           multiplier=multiplier,
                           norm=norm,
                           round_nearest=round_nearest)
    neck = GlobalAvgPooling()
    head = MultilayerDenseHead(576,
                               num_classes,
                               mid_channel=[1024],
                               activation=[nn.HSwish, None],
                               keep_prob=[1., 0.8]
                               )
    model = BaseClassifier(backbone, neck, head)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        arch = "mobilenet_v3_small"
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model


def mobilenet_v3_large(num_classes: int = 1000,
                       pretrained: bool = False,
                       multiplier: float = 1.0,
                       norm: Optional[nn.Cell] = None,
                       round_nearest: int = 8):
    """Mobilenet_v3 structure."""
    backbone = MobileNetV3(model_cfgs=model_cfgs['large'],
                           multiplier=multiplier,
                           norm=norm,
                           round_nearest=round_nearest)
    neck = GlobalAvgPooling()
    head = MultilayerDenseHead(960,
                               num_classes,
                               mid_channel=[1280],
                               activation=[nn.HSwish, None],
                               keep_prob=[1., 0.8]
                               )
    model = BaseClassifier(backbone, neck, head)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        arch = "mobilenet_v3_large"
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model
