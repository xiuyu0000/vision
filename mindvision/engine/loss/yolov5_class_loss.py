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
# ==============================================================================
"""Softmax Cross Entropy with Logits with label smoothing"""

from mindspore import nn
from mindspore import ops

from mindvision.engine.class_factory import ClassFactory, ModuleType

@ClassFactory.register(ModuleType.LOSS)
class YOLOv5ClassLoss(nn.Cell):
    """Loss for classification."""
    def __init__(self):
        super(YOLOv5ClassLoss, self).__init__()
        self.cross_entropy = ops.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, object_mask, predict_class, class_probs):
        class_loss = object_mask * self.cross_entropy(predict_class, class_probs)
        class_loss = self.reduce_sum(class_loss, ())
        return class_loss
