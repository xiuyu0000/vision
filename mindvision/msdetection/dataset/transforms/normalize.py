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
"""Detection Normalize."""

from mindspore.dataset.vision.c_transforms import Normalize

from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class DetectionNormalize:
    """
    Normalize the input image with respect to mean and standard deviation. This operator will normalize
    the input image with: output[channel] = (input[channel] - mean[channel]) / std[channel], where channel >= 1.
    """

    def __init__(self, mean, std):
        self.transform = Normalize(mean=mean, std=std)

    def __call__(self, image, boxes=None, labels=None):
        """
        Call method.
        """
        return self.transform(image), boxes, labels
