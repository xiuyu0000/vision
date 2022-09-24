# Copyright 2021
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
"""Mish activation."""

from mindspore import nn
from mindspore import ops


class Mish(nn.Cell):
    """
    Mish activation method:

    $$Mish = x * tanh(ln(1 + e^x))$$
    """

    def __init__(self):
        super(Mish, self).__init__()
        self.mul = ops.Mul()
        self.tanh = ops.Tanh()
        self.soft_plus = ops.Softplus()

    def construct(self, input_x):
        """ mish act construct. """
        res1 = self.soft_plus(input_x)
        tanh = self.tanh(res1)
        output = self.mul(input_x, tanh)
        return output
