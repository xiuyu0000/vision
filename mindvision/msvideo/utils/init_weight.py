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
"""Utility function for weight initialization. TODO: Why should add these? """

import math

import mindspore as msp
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.common import initializer as init
from mindspore.common.initializer import _assignment
from mindspore.common.initializer import _calculate_fan_in_and_fan_out
from mindspore.common.initializer import Normal, initializer, HeNormal, Zero

from mindvision.msvideo.models.blocks.unit3d import Unit3D


def init_weights(cell: nn.Cell, fc_init_std=0.01, zero_init_final_bn=True):
    """
    Performs ResNet style weight initialization.

    Args:
        fc_init_std (float): the expected standard deviation for fc layer.
        zero_init_final_bn (bool): if True, zero initialize the final bn for
            every bottleneck.

    Follow the initialization method proposed in:
    {He, Kaiming, et al.
    "Delving deep into rectifiers: Surpassing human-level
    performance on imagenet classification."
    arXiv preprint arXiv:1502.01852 (2015)}
    """
    for _, m in cell.cells_and_names():
        if isinstance(m, nn.Conv3d):
            m.weight.set_data(initializer(
                HeNormal(math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                m.weight.shape, m.weight.dtype))
            if m.bias is not None:
                m.bias.set_data(initializer(
                    Zero(), m.bias.shape, m.bias.dtype))
        elif isinstance(m, Unit3D):
            flag = False
            if (hasattr(m, "transform_final_bn")
                    and m.transform_final_bn and zero_init_final_bn):
                flag = True
            for _, n in m.cells_and_names():
                if isinstance(n, nn.BatchNorm3d):
                    if flag:
                        batchnorm_weight = 0.0
                    else:
                        batchnorm_weight = 1.0
                    if n.bn2d.gamma is not None:
                        fill = ops.Fill()
                        n.bn2d.gamma.set_data(fill(
                            msp.float32, n.bn2d.gamma.shape, batchnorm_weight))
                    if n.bn2d.beta is not None:
                        zeroslike = ops.ZerosLike()
                        n.bn2d.beta.set_data(zeroslike(n.bn2d.beta))

        if isinstance(m, nn.Dense):
            m.weight.set_data(initializer(
                Normal(sigma=fc_init_std, mean=0),
                shape=m.weight.shape, dtype=msp.float32))
            if m.bias is not None:
                zeroslike = ops.ZerosLike()
                m.bias.set_data(zeroslike(m.bias))


class UniformBias(init.Initializer):
    """bias uniform initializer"""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def _initialize(self, arr):
        fan_in, _ = _calculate_fan_in_and_fan_out(self.shape)
        bound = 1 / math.sqrt(fan_in)
        bound = Tensor(bound, msp.float32)
        data = ops.uniform(arr.shape, -bound, bound,
                           dtype=msp.float32).asnumpy()
        _assignment(arr, data)
