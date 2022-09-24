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
"""VGG network."""

from mindspore import nn

from mindvision.classification.models.backbones import VGG
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.models.head import MultilayerDenseHead
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel

__all__ = ['vgg11',
           'vgg13',
           'vgg16',
           'vgg19']


def _vgg(arch: str,
         batch_norm: bool,
         num_classes: int,
         pretrained: bool,
         ):
    """VGG architecture."""
    backbone = VGG(model_name=arch, batch_norm=batch_norm)
    head = MultilayerDenseHead(512 * 7 * 7,
                               num_classes,
                               mid_channel=[4096, 4096],
                               activation=[nn.ReLU, nn.ReLU, None],
                               keep_prob=[1., 0.5, 0.5]
                               )
    model = BaseClassifier(backbone, head)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model


def vgg11(num_classes: int = 1000,
          pretrained: bool = False,
          batch_norm: bool = False):
    """VGG11 network."""

    return _vgg(arch='vgg11',
                num_classes=num_classes,
                batch_norm=batch_norm,
                pretrained=pretrained)


def vgg13(num_classes: int = 1000,
          pretrained: bool = False,
          batch_norm: bool = False):
    """VGG13 network."""

    return _vgg(arch='vgg13',
                num_classes=num_classes,
                batch_norm=batch_norm,
                pretrained=pretrained)


def vgg16(num_classes: int = 1000,
          pretrained: bool = False,
          batch_norm: bool = False):
    """VGG16 network."""

    return _vgg(arch='vgg16',
                num_classes=num_classes,
                batch_norm=batch_norm,
                pretrained=pretrained)


def vgg19(num_classes: int = 1000,
          pretrained: bool = False,
          batch_norm: bool = False):
    """VGG19 network."""

    return _vgg(arch='vgg19',
                num_classes=num_classes,
                batch_norm=batch_norm,
                pretrained=pretrained)
