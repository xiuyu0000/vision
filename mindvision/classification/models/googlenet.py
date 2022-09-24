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
"""GoogLeNet."""

from mindvision.classification.models.backbones import GoogLeNet
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel

__all__ = ['_googlenet']


def _googlenet(num_classes: int = 10, num_channel: int = 1, pretrained: bool = False) -> GoogLeNet:
    backbone = GoogLeNet(num_classes=num_classes, in_channels=num_channel)
    model = BaseClassifier(backbone)

    if pretrained:
        arch = "googlenet"
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model
