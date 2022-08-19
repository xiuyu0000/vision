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
"""NLLLoss"""

from mindspore.nn.loss.loss import LossBase
from mindspore import ops
from mindspore.ops import functional as F
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ['NLLLoss']


@ClassFactory.register(ModuleType.LOSS)
class NLLLoss(LossBase):
    """NLLLoss"""

    def __init__(self, reduction="mean"):
        super(NLLLoss, self).__init__(reduction)
        self.one_hot = ops.OneHot()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, logits, labels):
        """NLLLoss construct."""
        label_one_hot = self.one_hot(labels, F.shape(logits)[-1], F.scalar_to_array(1.0), F.scalar_to_array(0.0))
        loss = self.reduce_sum(-1.0 * logits * label_one_hot, (1,))
        return self.get_loss(loss)
