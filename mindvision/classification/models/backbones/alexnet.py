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
"""AlexNet backbone."""

import mindspore.nn as nn
import mindspore.ops as ops

from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["AlexNet"]


@ClassFactory.register(ModuleType.BACKBONE)
class AlexNet(nn.Cell):
    """
    AlexNet backbone.

    Args:
        num_classes (int): The number of classes. Default: 10.
        num_channel (int): The number of input channels. Default: 3.
        keep_prob (float): Dropout keeping rate, between [0, 1]. Default: 0.5.
    """

    def __init__(self, num_classes=1000, num_channel=3, keep_prob=0.5):
        super(AlexNet, self).__init__()
        self.features = nn.SequentialCell([
            nn.Conv2d(
                num_channel,
                64,
                kernel_size=11,
                stride=4,
                padding=2,
                pad_mode="pad",
                has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid'),
            nn.Conv2d(
                64,
                192,
                kernel_size=5,
                padding=2,
                pad_mode="pad",
                has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid'),
            nn.Conv2d(
                192,
                384,
                kernel_size=3,
                padding=1,
                pad_mode="pad",
                has_bias=True),
            nn.ReLU(),
            nn.Conv2d(
                384,
                256,
                kernel_size=3,
                padding=1,
                pad_mode="pad",
                has_bias=True),
            nn.ReLU(),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                pad_mode="pad",
                has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')])
        self.avgpool = ops.AdaptiveAvgPool2D((6, 6))
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell([
            nn.Dropout(keep_prob),
            nn.Dense(256 * 6 * 6, 4096, has_bias=True),
            nn.ReLU(),
            nn.Dropout(keep_prob),
            nn.Dense(4096, 4096, has_bias=True),
            nn.ReLU(),
            nn.Dense(4096, num_classes, has_bias=True)])

    def construct(self, x):
        """
        AlexNet construct.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
