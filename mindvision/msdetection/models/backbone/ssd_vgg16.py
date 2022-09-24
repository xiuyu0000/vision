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
"""VGG16 backbone for SSD."""

import mindspore.nn as nn

from mindvision.classification.models.backbones import VGG16
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["SSDVGG16"]


@ClassFactory.register(ModuleType.BACKBONE)
class SSDVGG16(nn.Cell):
    """
    VGG16 backbone for SSD.
    """

    def __init__(self):
        super(SSDVGG16, self).__init__()

        # VGG16 backbone: block1~5
        net = VGG16()
        self.features = nn.CellList(list(net.features))

        # SSD blocks: block6~7
        self.b6_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6, pad_mode='pad')
        self.b6_2 = nn.Dropout(0.5)

        self.b7_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        self.b7_2 = nn.Dropout(0.5)

        # Extra Feature Layers: block8~11
        self.b8_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, padding=1, pad_mode='pad')
        self.b8_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, pad_mode='valid')

        self.b9_1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=1, pad_mode='pad')
        self.b9_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, pad_mode='valid')

        self.b10_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.b10_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, pad_mode='valid')

        self.b11_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.b11_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, pad_mode='valid')

    def construct(self, x):
        """
        Forward pass.
        """
        # VGG16 backbone: block1~5
        features = ()

        for block in self.features:
            x = block(x)
            features = features + (x,)

        block4, x = features[-9], features[-1]

        # SSD blocks: block6~7
        x = self.b6_1(x)  # 1024
        x = self.b6_2(x)

        x = self.b7_1(x)  # 1024
        x = self.b7_2(x)
        block7 = x

        # Extra Feature Layers: block8~11
        x = self.b8_1(x)  # 256
        x = self.b8_2(x)  # 512
        block8 = x

        x = self.b9_1(x)  # 128
        x = self.b9_2(x)  # 256
        block9 = x

        x = self.b10_1(x)  # 128
        x = self.b10_2(x)  # 256
        block10 = x

        x = self.b11_1(x)  # 128
        x = self.b11_2(x)  # 256
        block11 = x

        # boxes
        multi_feature = (block4, block7, block8, block9, block10, block11)

        return multi_feature
