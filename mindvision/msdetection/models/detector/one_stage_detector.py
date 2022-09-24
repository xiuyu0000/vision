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
"""One stage detector."""

from mindvision.msdetection.models.builder import build_backbone, build_neck, build_head
from mindvision.msdetection.models.detector.base_detector import BaseDetector
from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.DETECTOR)
class OneStageDetector(BaseDetector):
    """
    Base Class of one-stage detector.
    """

    def __init__(self, backbone, bbox_head, neck=None):
        """Constructor for OneStageDetector"""
        super(OneStageDetector, self).__init__()
        self.backbone = build_backbone(backbone) if isinstance(backbone, dict) else backbone

        if neck:
            self.neck = build_neck(neck) if isinstance(neck, dict) else neck

        self.bbox_head = build_head(bbox_head) if isinstance(bbox_head, dict) else bbox_head

    def construct(self, images, boxes, labels):
        """Construct model."""
        x = self.backbone(images)

        if self.has_neck:
            x = self.neck(x)

        if self.training:
            x = self.bbox_head.construct_train(x, boxes, labels)
        else:
            x = self.bbox_head.construct_eval(x)
        return x
