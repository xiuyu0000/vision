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
"""Point cloud transforms functions."""

import numpy as np
import mindspore.dataset.transforms.py_transforms as trans
from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class PcScale(trans.PyTensorOperation):
    """
    Random scale the input point cloud with: output = input * scale

    Args:
       scale_low(float): low bound for the scale size.
       scale_high(float): upper bound for the scale size.

    Examples:
      >>> #  Random scale the input point cloud in type numpy.
      >>> trans = [PcScale(scale_low=0.8, scale_high=1.2)]
   """

    def __init__(self, scale_low=0.8, scale_high=1.2):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, x):
        """
        Args:
           Point cloud(numpy array): point cloud data.

        Returns:
           transformed Point cloud: point cloud data.
        """
        if isinstance(x, np.ndarray):
            scale = np.random.uniform(self.scale_low, self.scale_high)
            output = x * scale
            return output
        raise AssertionError(
            "Type of input should be numpy but got {}.".format(
                type(x).__name__))
