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
"""Point2ClsHead"""

from typing import List
from mindspore import nn
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ['Point2ClsHead']


@ClassFactory.register(ModuleType.HEAD)
class Point2ClsHead(nn.Cell):
    """
    Point2ClsHead architecture.

    Args:
        input_channel (int): The number of input channel.
        num_classes (int): Number of classes.
        mid_channel (list): Number of channels in the hidden fc layers.
        keep_prob (list): Dropout keeping rate, between [0, 1]. E.g. rate=0.9, means dropping out 10% of
        input.

    Returns:
        Tensor, output tensor.

    Example:
        >>data = Tensor(np.random.randn(32, 1024), dtype=mindspore.float32)
        >>model = ClsHead(input_channel=1024, num_classes=40, mid_channel=[512, 256], keep_prob=0.4)
        >>predict = model(data)
        >>print(predict.shape)
    """

    def __init__(self, input_channel: int, num_classes: int, mid_channel: List[int], keep_prob: List[float]):
        super(Point2ClsHead, self).__init__()

        head = []
        length = len(mid_channel)
        for i in range(length):
            head.append(nn.Dense(input_channel, mid_channel[i]))
            head.append(nn.BatchNorm1d(mid_channel[i]))
            head.append(nn.ReLU())
            head.append(nn.Dropout(keep_prob))
            input_channel = mid_channel[i]

        self.classifier = nn.SequentialCell(head)
        self.fc = nn.Dense(mid_channel[-1], num_classes)
        self.logsoftmax = nn.LogSoftmax(axis=1)

    def construct(self, x):
        """Point2ClsHead construct"""
        x = self.classifier(x)
        x = self.fc(x)
        x = self.logsoftmax(x)
        return x
