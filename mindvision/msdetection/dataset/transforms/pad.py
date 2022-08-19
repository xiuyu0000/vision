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
# ==============================================================================
"""Detection Pad."""

import numpy as np

from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class DetectionPad:
    """
    Pad the boxes and labels.
    """

    def __init__(self, pad_max_number=128):
        self.pad_max_number = pad_max_number

    def __call__(self, image, boxes=None, labels=None):
        """
        Call method.
        """
        boxes = np.pad(np.array(boxes),
                       ((0, self.pad_max_number - len(boxes)), (0, 0)),
                       mode="constant",
                       constant_values=0)
        labels = np.pad(np.array(labels),
                        ((0, self.pad_max_number - len(labels)),),
                        mode="constant",
                        constant_values=-1)
        return image, boxes, labels
