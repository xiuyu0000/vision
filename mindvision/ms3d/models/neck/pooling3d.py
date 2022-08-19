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
"""3D Pooling neck."""

from mindspore import nn

from mindspore import ops
from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.NECK)
class GlobalAvgPooling3D(nn.Cell):
    """
    A module of Global average pooling for 3D features.

    Args:
        keep_dims (bool): Specifies whether to keep dimension shape the same as input feature.
            E.g. `True`. Default: False

    Returns:
        Tensor, output tensor.
    """

    def __init__(self,
                 keep_dims: bool = True
                 ) -> None:
        super(GlobalAvgPooling3D, self).__init__()
        self.mean = ops.ReduceMean(keep_dims=keep_dims)

    def construct(self, x):
        x = self.mean(x, (2, 3, 4))
        return x
