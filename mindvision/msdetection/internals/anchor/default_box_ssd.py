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
"""Generate default box."""

from typing import List
import math
import itertools
import numpy as np

from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["GenerateDefaultBoxes"]


@ClassFactory.register(ModuleType.ANCHOR_GENERATOR)
class GenerateDefaultBoxes:
    """
    Generate Default boxes for SSD, follows the order of (W, H, archor_sizes).
    `self.default_boxes` has a shape of [archor_sizes, H, W, 4], the last dimension is [y, x, h, w].
    `self.default_boxes_tlbr` has a shape as `self.default_boxes`, the last dimension is [y1, x1, y2, x2].
    """

    def __init__(self,
                 img_shape: int,
                 steps: List[int],
                 max_scale: float,
                 min_scale: float,
                 num_default: List[int],
                 feature_size: List[int],
                 aspect_ratios: List[List[float]]
                 ):
        fk = img_shape[0] / np.array(steps)
        scale_rate = (max_scale - min_scale) / (len(num_default) - 1)
        scales = [min_scale + scale_rate * i for i in range(len(num_default))] + [1.0]
        self.default_boxes = []
        for idx, feature in enumerate(feature_size):
            sk1 = scales[idx]
            sk2 = scales[idx + 1]
            sk3 = math.sqrt(sk1 * sk2)
            if idx == 0 and not aspect_ratios[idx]:
                w, h = sk1 * math.sqrt(2), sk1 / math.sqrt(2)
                all_sizes = [(0.1, 0.1), (w, h), (h, w)]
            else:
                all_sizes = [(sk1, sk1)]
                for aspect_ratio in aspect_ratios[idx]:
                    w, h = sk1 * math.sqrt(aspect_ratio), sk1 / math.sqrt(aspect_ratio)
                    all_sizes.append((w, h))
                    all_sizes.append((h, w))
                all_sizes.append((sk3, sk3))

            for i, j in itertools.product(range(feature), repeat=2):
                for w, h in all_sizes:
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append([cx, cy, w, h])

        def to_tlbr(cx, cy, w, h):
            return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

        # For IoU calculation
        self.default_boxes_tlbr = np.array(tuple(to_tlbr(*i) for i in self.default_boxes), dtype='float32')
        self.default_boxes = np.array(self.default_boxes, dtype='float32')
