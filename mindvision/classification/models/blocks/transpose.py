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
"""Transpose layer."""

from mindspore import ops, nn


class TransposeChannel(nn.Cell):
    """
    Transpose data's channel axis from channel_first(channel_last) to channel_last(channel_first).

    Args:
        target(str): 'channel_first' or 'channel_last'.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> transpose = TransposeChannel(target='channel_first')
    """

    def __init__(self, target='channel_first'):
        super(TransposeChannel, self).__init__()
        self.transpose = ops.Transpose()
        if target == 'channel_first':
            self.perm = (0, 3, 1, 2)
        elif target == 'channel_last':
            self.perm = (0, 2, 3, 1)

    def construct(self, x):
        """Transpose layer construct"""
        x = self.transpose(x, self.perm)
        return x
