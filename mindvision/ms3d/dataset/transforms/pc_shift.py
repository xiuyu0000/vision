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
class PcShift(trans.PyTensorOperation):
    """
    Random shift the input point cloud with: output = input + scale

    Args:
       shift_range(float): min and max value for the shift distance.

    Examples:
      >>> #  Random shift the input point cloud in type numpy.
      >>> trans = [transform.PcShift(shift_range=0.1)]
   """

    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, x):
        """
        Args:
           Point cloud(numpy array): point cloud data.

        Returns:
           transformed Point cloud: point cloud data.
        """
        if isinstance(x, np.ndarray):
            shift_range = np.random.uniform(-self.shift_range, self.shift_range, 3)
            output = x + shift_range
            return output
        raise AssertionError(
            "Type of input should be numpy but got {}.".format(
                type(x).__name__))
