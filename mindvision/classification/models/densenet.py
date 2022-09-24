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
"""Densenet."""
from typing import List
from mindvision.classification.models.backbones import DenseNet
from mindvision.classification.models.classifiers.base import BaseClassifier
from mindvision.classification.models.head import DenseHead
from mindvision.classification.models.neck.pooling import GlobalAvgPooling
from mindvision.utils.load_pretrained_model import LoadPretrainedModel
from mindvision.classification.utils.model_urls import model_urls


__all__ = [
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'densenet232',
    'densenet264'
]


def densenet(arch: str,
             growth_rate: int,
             block_config: List[int],
             num_init_features: int,
             num_classes: int,
             pretrained: bool
             ) -> DenseNet:
    """ResNet architecture."""
    backbone = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_init_features=num_init_features
    )
    neck = GlobalAvgPooling()
    head = DenseHead(
        input_channel=backbone.get_out_channels(),
        num_classes=num_classes,
    )
    model = BaseClassifier(backbone, neck, head)
    if pretrained:
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model


def densenet121(num_classes: int = 1000,
                pretrained: bool = False,
                ) -> DenseNet:
    return densenet("densenet121", 32, [6, 12, 24, 16], 64, num_classes, pretrained)


def densenet161(num_classes: int = 1000,
                pretrained: bool = False,
                ) -> DenseNet:
    return densenet("densenet161", 48, [6, 12, 36, 24], 96, num_classes, pretrained)


def densenet169(num_classes: int = 1000,
                pretrained: bool = False,
                ) -> DenseNet:
    return densenet("densenet169", 32, [6, 12, 32, 32], 64, num_classes, pretrained)


def densenet201(num_classes: int = 1000,
                pretrained: bool = False,
                ) -> DenseNet:
    return densenet("densenet201", 32, [6, 12, 48, 32], 64, num_classes, pretrained)


def densenet232(num_classes: int = 1000,
                pretrained: bool = False,
                ) -> DenseNet:
    return densenet("densenet232", 48, [6, 32, 48, 48], 96, num_classes, pretrained)


def densenet264(num_classes: int = 1000,
                pretrained: bool = False,
                ) -> DenseNet:
    return densenet("densenet264", 32, [6, 12, 64, 48], 64, num_classes, pretrained)
