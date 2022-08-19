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
"""S3FD backbone, based on VGG16"""

from mindspore import nn
from mindspore import ops
from mindvision.classification.models.backbones import VGG16
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["S3FDBackbone"]


@ClassFactory.register(ModuleType.BACKBONE)
class S3FDBackbone(nn.Cell):
    """
    S3FD backbone implementation.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        detection layers output: [f3_3, f4_3, f5_3, ffc7, f6_2, f7_2]

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindvision.msdetection.models.backbone.s3fd import S3FDBackbone
        >>> net = S3FDBackbone()
        >>> x = ms.Tensor(np.ones([1, 3, 303, 240]), ms.float32)
        >>> print(len(net(x)))
        6
    """

    def __init__(self):
        super(S3FDBackbone, self).__init__()
        # Base Convolutional Layers(VGG16 through Pool5 layer)
        net = VGG16()
        self.features = nn.CellList(list(net.features))

        # Extra Convolutional Layers
        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, pad_mode="pad", padding=3, has_bias=True)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, pad_mode="pad", padding=0, has_bias=True)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, pad_mode="pad", padding=0, has_bias=True)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, pad_mode="pad", padding=1, has_bias=True)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, pad_mode="pad", padding=0, has_bias=True)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, pad_mode="pad", padding=1, has_bias=True)

    def construct(self, x):
        """S3FD backbone construct."""
        features = ()
        for block in self.features:
            x = block(x)
            features = features + (x,)

        f3_3 = features[15]
        f4_3 = features[22]
        f4_3 = features[29]

        h = ops.ReLU()(self.fc6(x))
        h = ops.ReLU()(self.fc7(h))
        ffc7 = h
        h = ops.ReLU()(self.conv6_1(h))
        h = ops.ReLU()(self.conv6_2(h))
        f6_2 = h
        h = ops.ReLU()(self.conv7_1(h))
        h = ops.ReLU()(self.conv7_2(h))
        f7_2 = h

        return [f3_3, f4_3, f4_3, ffc7, f6_2, f7_2]
