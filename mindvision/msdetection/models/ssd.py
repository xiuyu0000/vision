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
"""SSD network."""

from mindvision.msdetection.models.backbone import SSDMobileNetV2
from mindvision.msdetection.models.neck import SSDMobileNetV2Neck
from mindvision.msdetection.models.head import MultiBox
from mindvision.msdetection.internals.anchor import GenerateDefaultBoxes
from mindvision.msdetection.models.detector import OneStageDetector

from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel

__all__ = ['ssd_mobilenet_v2']


def ssd_mobilenet_v2(num_classes: int = 81,
                     pretrained: bool = False):
    """
    MobileNetV2 as SSD backbone.
    """
    backbone = SSDMobileNetV2()
    neck = SSDMobileNetV2Neck(extras_in_channels=[256, 576, 1280, 512, 256, 256],
                              extras_out_channels=[576, 1280, 512, 256, 256, 128],
                              extras_strides=[1, 1, 2, 2, 2, 2],
                              extras_ratios=[0.2, 0.2, 0.2, 0.25, 0.5, 0.25])
    anchor_generator = GenerateDefaultBoxes(img_shape=(300, 300),
                                            steps=[16, 32, 64, 100, 150, 300],
                                            max_scale=0.95,
                                            min_scale=0.2,
                                            num_default=[3, 6, 6, 6, 6, 6],
                                            feature_size=[19, 10, 5, 3, 2, 1],
                                            aspect_ratios=[[], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
                                            )
    bbox_head = MultiBox(num_classes=num_classes,
                         extras_out_channels=[576, 1280, 512, 256, 256, 128],
                         num_default=[3, 6, 6, 6, 6, 6],
                         num_ssd_boxes=1917,
                         gamma=2.0,
                         alpha=0.75,
                         anchor_generator=anchor_generator,
                         prior_scaling=[0.1, 0.2]
                         )
    model = OneStageDetector(backbone=backbone,
                             bbox_head=bbox_head,
                             neck=neck)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        arch = "ssd_mobilenet_v2"
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model
