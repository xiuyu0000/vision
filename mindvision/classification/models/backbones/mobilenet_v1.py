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

from mindvision.classification.models.blocks import ConvNormActivation
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["MobileNetV1"]


@ClassFactory.register(ModuleType.BACKBONE)
class MobileNetV1(nn.Cell):
    """
    MobileNet V1 backbone.
    """

    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.layers = [
            ConvNormActivation(3, 32, 3, 2, activation=nn.ReLU6),  # Conv0

            ConvNormActivation(32, 32, 3, 1, groups=32, activation=nn.ReLU6),  # Conv1_depthwise
            ConvNormActivation(32, 64, 1, 1, activation=nn.ReLU6),  # Conv1_pointwise
            ConvNormActivation(64, 64, 3, 2, groups=64, activation=nn.ReLU6),  # Conv2_depthwise
            ConvNormActivation(64, 128, 1, 1, activation=nn.ReLU6),  # Conv2_pointwise

            ConvNormActivation(128, 128, 3, 1, groups=128, activation=nn.ReLU6),  # Conv3_depthwise
            ConvNormActivation(128, 128, 1, 1, activation=nn.ReLU6),  # Conv3_pointwise
            ConvNormActivation(128, 128, 3, 2, groups=128, activation=nn.ReLU6),  # Conv4_depthwise
            ConvNormActivation(128, 256, 1, 1, activation=nn.ReLU6),  # Conv4_pointwise

            ConvNormActivation(256, 256, 3, 1, groups=256, activation=nn.ReLU6),  # Conv5_depthwise
            ConvNormActivation(256, 256, 1, 1, activation=nn.ReLU6),  # Conv5_pointwise
            ConvNormActivation(256, 256, 3, 2, groups=256, activation=nn.ReLU6),  # Conv6_depthwise
            ConvNormActivation(256, 512, 1, 1, activation=nn.ReLU6),  # Conv6_pointwise

            ConvNormActivation(512, 512, 3, 1, groups=512, activation=nn.ReLU6),  # Conv7_depthwise
            ConvNormActivation(512, 512, 1, 1, activation=nn.ReLU6),  # Conv7_pointwise
            ConvNormActivation(512, 512, 3, 1, groups=512, activation=nn.ReLU6),  # Conv8_depthwise
            ConvNormActivation(512, 512, 1, 1, activation=nn.ReLU6),  # Conv8_pointwise
            ConvNormActivation(512, 512, 3, 1, groups=512, activation=nn.ReLU6),  # Conv9_depthwise
            ConvNormActivation(512, 512, 1, 1, activation=nn.ReLU6),  # Conv9_pointwise
            ConvNormActivation(512, 512, 3, 1, groups=512, activation=nn.ReLU6),  # Conv10_depthwise
            ConvNormActivation(512, 512, 1, 1, activation=nn.ReLU6),  # Conv10_pointwise
            ConvNormActivation(512, 512, 3, 1, groups=512, activation=nn.ReLU6),  # Conv11_depthwise
            ConvNormActivation(512, 512, 1, 1, activation=nn.ReLU6),  # Conv11_pointwise

            ConvNormActivation(512, 512, 3, 2, groups=512, activation=nn.ReLU6),  # Conv12_depthwise
            ConvNormActivation(512, 1024, 1, 1, activation=nn.ReLU6),  # Conv12_pointwise
            ConvNormActivation(1024, 1024, 3, 1, groups=1024, activation=nn.ReLU6),  # Conv13_depthwise
            ConvNormActivation(1024, 1024, 1, 1, activation=nn.ReLU6),  # Conv13_pointwise
        ]

        self.features = nn.SequentialCell(self.layers)

    def construct(self, x):
        """Forward pass"""
        output = self.features(x)
        return output
