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

from mindspore import nn
from mindspore import ops

__all__ = ['Swish']


class Swish(nn.Cell):
    """
    "Swish activation function: x * sigmoid(x).

    Args:
        None

    Return:
        Tensor

    Example:
        >>> x = Tensor(((20, 16), (50, 50)), mindspore.float32)
        >>> Swish()(x)
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.result = None

    def construct(self, x):
        """ construct swish """
        sigmoid = ops.Sigmoid()
        result = x * sigmoid(x)
        return result

    def bprop(self, x, dout):
        """ bprop """
        sigmoid = ops.Sigmoid()
        sigmoid_x = sigmoid(x)
        result = dout * (sigmoid_x * (1 + x * (1 - sigmoid_x)))
        return result
