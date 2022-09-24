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
"""BaseClassifier"""

from mindspore import nn

from mindvision.classification.models.builder import build_backbone, build_neck, build_head, build_embed
from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.RECOGNIZER)
class BaseRecognizer(nn.Cell):
    """
    Generate recognizer for video recognition task.
    """

    def __init__(self, backbone, embedding=None, neck=None, head=None):
        super(BaseRecognizer, self).__init__()

        self.backbone = build_backbone(backbone) if isinstance(backbone, dict) else backbone

        if embedding:
            self.embed = build_embed(embedding) if isinstance(embedding, dict) else embedding
            self.with_embed = True
        else:
            self.with_embed = False
        if neck:
            self.neck = build_neck(neck) if isinstance(neck, dict) else neck
            self.with_neck = True
        else:
            self.with_neck = False
        if head:
            self.head = build_head(head) if isinstance(head, dict) else head
            self.with_head = True
        else:
            self.with_head = False

    def construct(self, x):
        """construct for video base recognizer."""
        if self.with_embed:
            x = self.embed(x)

        x = self.backbone(x)

        if self.with_neck:
            x = self.neck(x)
        if self.with_head:
            x = self.head(x)

        return x
