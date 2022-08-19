# Copyright 2021 Huawei Technologies Co., Ltd
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
"""GoogLeNet backbone."""

from mindspore import nn

from mindvision.classification.models.blocks import Inception

from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["GoogLeNet"]


@ClassFactory.register(ModuleType.BACKBONE)
class GoogLeNet(nn.Cell):
    """
    Args:
        num_classes (int): class number
        in_channels (float): channel number
    Returns:
        Tensor, output tensor.
    """

    def __init__(self, num_classes=10, in_channels=1):
        super(GoogLeNet, self).__init__()
        self.model = nn.SequentialCell([
            nn.Conv2d(in_channels,
                      out_channels=64,
                      kernel_size=7,
                      stride=2
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=192,
                      kernel_size=3
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same'),
            Inception(192, c1=64, c2=[96, 128], c3=[16, 32], c4=32),
            Inception(256, c1=128, c2=[128, 192], c3=[32, 96], c4=64),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same'),
            Inception(480, c1=192, c2=[96, 208], c3=[16, 48], c4=64),
            Inception(512, c1=160, c2=[112, 224], c3=[24, 64], c4=64),
            Inception(512, c1=128, c2=[128, 256], c3=[24, 64], c4=64),
            Inception(512, c1=112, c2=[144, 288], c3=[32, 64], c4=64),
            Inception(528, c1=256, c2=[160, 320], c3=[32, 128], c4=128),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same'),
            Inception(832, c1=256, c2=[160, 320], c3=[32, 128], c4=128),
            Inception(832, c1=384, c2=[192, 384], c3=[48, 128], c4=128),
            nn.AvgPool2d(kernel_size=7, stride=1, pad_mode='same'),
            nn.Dropout(keep_prob=0.4),
            nn.Flatten(),
            nn.Dense(in_channels=50176, out_channels=num_classes),
            nn.Softmax(axis=1)
        ])

    def construct(self, x):
        for layer in self.model:
            x = layer(x)
        return x
