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
"""Detection HWC2CHW."""

from mindspore.dataset.vision.c_transforms import HWC2CHW

from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class DetectionHWC2CHW:
    """
    Transpose the input image from shape <H, W, C> to shape <C, H, W>. The input image should be 3 channels image.
    """

    def __init__(self):
        self.transform = HWC2CHW()

    def __call__(self, image, boxes=None, labels=None):
        """
        Call method.
        """
        return self.transform(image), boxes, labels
