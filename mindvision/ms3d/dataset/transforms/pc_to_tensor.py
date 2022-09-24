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
"""Point cloud transforms functions."""

import numpy as np
import mindspore.dataset.transforms.py_transforms as trans
from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class PcToTensor(trans.PyTensorOperation):
    """
    Convert the input point cloud in type numpy.ndarray of shape (N, C)
    to numpy.ndarray of shape (C, N).

    Args:
       new_order(tuple), new_order of output.

    Examples:
      >>> #  Convert the input video frames in type numpy
      >>> trans = [transform.PcToTensor()]
   """

    def __init__(self, order=(1, 0)):
        self.order = tuple(order)

    def __call__(self, x):
        """
        Args:
           Video(list): Video to be tensor.

        Returns:
           seq video: Tensor of seq video.
        """
        if isinstance(x, np.ndarray):
            return np.transpose(x, self.order).astype(np.float32)
        raise AssertionError(
            "Type of input should be numpy but got {}.".format(
                type(x).__name__))
