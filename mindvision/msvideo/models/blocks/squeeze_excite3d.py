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
"""Custom operators."""

from typing import Union

from mindspore import nn

from mindvision.classification.engine.ops.swish import Swish
from mindvision.classification.models.utils import make_divisible
from mindvision.msvideo.engine.ops.adaptiveavgpool3d import AdaptiveAvgPool3D


class SqueezeExcite3D(nn.Cell):
    """
    Squeeze-and-Excitation (SE) block implementation.

    Args:
        dim_in (int): the channel dimensions of the input.
        ratio (float): the channel reduction ratio for squeeze.
        act_fn (Union[str, nn.Cell]): the activation of conv_expand: Default: Swish.

    Returns:
        Tensor
    """
    def __init__(self, dim_in, ratio, act_fn: Union[str, nn.Cell] = Swish):
        super(SqueezeExcite3D, self).__init__()
        self.avg_pool = AdaptiveAvgPool3D((1, 1, 1))
        v = dim_in * ratio
        dim_fc = make_divisible(v=v, divisor=8)
        self.fc1 = nn.Conv3d(dim_in, dim_fc, 1, has_bias=True)
        self.fc1_act = nn.ReLU() if act_fn else Swish()
        self.fc2 = nn.Conv3d(dim_fc, dim_in, 1, has_bias=True)
        self.fc2_sig = nn.Sigmoid()

    def construct(self, x):
        x_in = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc2(x)
        x = self.fc2_sig(x)
        return x_in * x
