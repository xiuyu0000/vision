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
"""MobileNet_v2 backbone."""

from mindspore import nn

from mindvision.classification.models.blocks import ConvNormActivation
from mindvision.classification.models.utils import make_divisible
from mindvision.classification.models.backbones import MobileNetV2
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["SSDMobileNetV2"]


@ClassFactory.register(ModuleType.BACKBONE)
class SSDMobileNetV2(nn.Cell):
    """
    MobileNet V2 backbone.
    """

    def __init__(self):
        super(SSDMobileNetV2, self).__init__()
        net = MobileNetV2()
        features = net.features
        self.features_1 = features[:14]
        self.features_2 = features[14:]

        input_channel = make_divisible(96 * 1.0, 8)
        hidden_channel = int(round(input_channel * 6))
        self.expand_layer = ConvNormActivation(input_channel, hidden_channel, kernel_size=1, activation=nn.ReLU6)

    def construct(self, x):
        """Forward pass"""
        out = self.features_1(x)
        expand_layer = self.expand_layer(out)
        out = self.features_2(out)

        return expand_layer, out
