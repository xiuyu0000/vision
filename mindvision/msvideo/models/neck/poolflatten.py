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
"""Avgpooling flatten neck"""

from typing import Optional, Union, List, Tuple

from mindspore import nn

from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.msvideo.engine.ops import AvgPool3D, AdaptiveAvgPool3D


@ClassFactory.register(ModuleType.NECK)
class AvgpoolFlatten(nn.Cell):
    """
    Avgpooling and flatten

    Args:
        pool_size (Optional[Union[List[int], Tuple[int]]]): a single entry list of kernel size for
            spatiotemporal pooling for the TxHxW dimensions. Default: None.

    Inputs:
        x(Tensor): The input Tensor in the form of :math:`(N, C, D_{in}, H_{in}, W_{in})`.

    Returns:
        Tensor, the pooled and flattened Tensor
    """

    def __init__(self,
                 pool_size: Optional[Union[List[int], Tuple[int]]] = None):
        super(AvgpoolFlatten, self).__init__()
        if pool_size is None:
            self.avg_pool = AdaptiveAvgPool3D((1, 1, 1))
        else:
            self.avg_pool = AvgPool3D(tuple(pool_size))

        self.flatten = nn.Flatten()

    def construct(self, x):
        """Avgpooling Flatten construct."""
        x = self.avg_pool(x)
        x = self.flatten(x)
        return x
