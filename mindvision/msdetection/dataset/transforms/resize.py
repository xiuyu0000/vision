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
"""Detection Resize."""

from mindspore.dataset.vision.c_transforms import Resize
from mindspore.dataset.vision import Inter

from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class DetectionResize:
    """
    Resize the input image to the given size with a given interpolation mode.
    """

    def __init__(self, size, interpolation=Inter.LINEAR):
        self.transform = Resize(size=size, interpolation=interpolation)

    def __call__(self, image, boxes=None, labels=None):
        """
        Call method.
        """
        return self.transform(image), boxes, labels
