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
"""Tracking head for VisTR"""

from mindspore import nn

from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.classification.models.head.multilayer_dense_head import MultilayerDenseHead


@ClassFactory.register(ModuleType.HEAD)
class VisTrackingHead(nn.Cell):
    """
    Tracking head for VIS Transformer.

    Args:
        in_channels (int): Number of channels in input feature.
        num_classes (int): Number of classes to be classified.

    Inputs:
        Tensor of shape (B, 384).

    Returns:
        A dict of
            {'class_logits': outputs_class,
             'boxes_coord': outputs_coord}.
    """

    def __init__(self,
                 input_channel: int = 384,
                 num_classes: int = 10,
                 ):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.fc_cls = nn.Dense(in_channels=input_channel, out_channels=num_classes + 1)
        self.mlp_bbox = MultilayerDenseHead(
            input_channel=input_channel,
            num_classes=4,
            mid_channel=[input_channel, input_channel],
            keep_prob=[1., 1., 1.],
            activation=['relu', 'relu', None]
        )

    def construct(self, x):
        outputs_class = self.fc_cls(x)
        outputs_coord = self.sigmoid(self.mlp_bbox(x))
        out = {'class_logits': outputs_class[-1],
               'boxes_coord': outputs_coord[-1]}
        return out
