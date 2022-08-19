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
"""MobileNet_v1 backbone."""

from mindspore import nn

from mindvision.classification.models.backbones import MobileNetV1
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["SSDMobileNetV1"]


@ClassFactory.register(ModuleType.BACKBONE)
class SSDMobileNetV1(nn.Cell):
    """
    MobileNet V1 backbone.
    """

    def __init__(self):
        super(SSDMobileNetV1, self).__init__()
        backbone = MobileNetV1()
        layers = backbone.layers
        self.features = nn.CellList(layers)

    def construct(self, x):
        """Forward pass"""
        features = ()
        for block in self.features:
            x = block(x)
            features = features + (x,)

        return features
