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
""" ARN neck(mainly second-order pooling with power normalization)."""

from mindspore import nn
from mindspore import ops

from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.NECK)
class ARNNeck(nn.Cell):
    """
    ARN neck architecture.

    Args:
        class_num (int): Number of classes. Default: 5.
        support_num_per_class (int): Number of samples in support set per class. Default: 5.
        sigma: Controls the slope of PN. Default: 100.

    Returns:
        Tensor, output 2 tensors.
    """

    def __init__(self,
                 class_num: int = 5,
                 support_num_per_class: int = 5,
                 sigma: int = 100
                 ):
        super(ARNNeck, self).__init__()
        self.class_num = class_num
        self.support_num_per_class = support_num_per_class
        self.sigma = sigma

        self.mm = ops.MatMul(transpose_b=True)
        self.sigmoid = ops.Sigmoid()
        self.mean = ops.ReduceMean()
        self.expand = ops.ExpandDims()
        self.stack_feature = ops.Stack(axis=0)

    def power_norm(self, x):
        """
        Define the operation of Power Normalization.

        Args:
            x (Tensor): Tensor of shape :math:`(C_{in}, C_{in})`.

        Returns:
            Tensor of shape: math:`(C_{out}, C_{out})`.
        """
        out = 2.0*self.sigmoid(self.sigma*x) - 1.0
        return out

    def construct(self, support_features, query_features):
        """test construct of arn neck"""
        channel = support_features.shape[1]
        so_support_features = []
        so_query_features = []

        for dd in range(support_features.shape[0]):
            s = support_features[dd, :, :].reshape(channel, -1)
            s = (1.0 / s.shape[1]) * self.mm(s, s)
            so_support_features.append(self.power_norm(s / s.trace()))
        so_support_features = self.stack_feature(so_support_features)

        for dd in range(query_features.shape[0]):
            t = query_features[dd, :, :].view(channel, -1)
            t = (1.0 / t.shape[1]) * self.mm(t, t)
            so_query_features.append(self.power_norm(t / t.trace()))
        so_query_features = self.stack_feature(so_query_features)

        so_support_features = so_support_features.reshape(
            self.class_num, self.support_num_per_class, 1, channel, channel).mean(1)  # z-shot, average
        so_query_features = self.expand(so_query_features, 1)  # 1 * C * C

        return so_support_features, so_query_features
