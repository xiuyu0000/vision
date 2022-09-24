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
# ==============================================================================
"""Detection Random color adjust."""

from mindspore.dataset.vision.c_transforms import RandomColorAdjust

from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class DetectionRandomColorAdjust:
    """
    Randomly adjust the brightness, contrast, saturation, and hue of the input image.
    """

    def __init__(self, brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0)):
        self.transform = RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image, boxes=None, labels=None):
        """
        Call method.
        """
        return self.transform(image), boxes, labels
