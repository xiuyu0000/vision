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
"""Smooth l1 loss. TODO: merge the loss into mindspore.nn.loss and clean these folder."""

import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import ops

from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.LOSS)
class SmoothL1Loss(nn.Cell):
    """Loss for x and y."""

    def __init__(self, beta=1.0, reduction="mean", index=(), loss_weight=1):
        super(SmoothL1Loss, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.smooth_l1_loss = ops.SmoothL1Loss(beta=beta)
        self.index = index

    def construct(self, predict, groud_truth, weight, normalizer):
        loss = self.smooth_l1_loss(predict, groud_truth)
        loss = loss * weight
        loss = ops.Cast()(loss, mstype.float32)
        if self.reduction == "mean":
            loss = self.reduce_sum(loss, self.index) / normalizer
        elif self.reduction == "sum":
            loss = self.reduce_sum(loss, self.index)
        loss = loss * self.loss_weight
        return loss
