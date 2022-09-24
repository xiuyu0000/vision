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
"""AlexNet"""

from mindvision.classification.models.backbones import AlexNet
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel

__all__ = ['alex']


def alex(num_classes: int = 1000,
         num_channel: int = 3,
         dropout: float = 0.5,
         pretrained: bool = False,
         ) -> AlexNet:
    """AlexNet architecture."""
    backbone = AlexNet(
        num_classes=num_classes,
        num_channel=num_channel,
        keep_prob=1 - dropout
    )
    model = BaseClassifier(backbone)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        arch = "alexnet"
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model
