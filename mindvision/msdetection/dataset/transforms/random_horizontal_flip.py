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
"""Detection Random horizontal flip."""

import random
import cv2

from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class DetectionRandomHorizontalFlip:
    """
    Randomly flip the input image horizontally with a given probability.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes=None, labels=None):
        """
        Call method.
        """
        if random.random() < self.prob:
            image = cv2.flip(image, 1, dst=None)
            boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
        return image, boxes, labels
