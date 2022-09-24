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
"""i2p module."""

from mindspore import nn
from mindspore import ops
from mindvision.classification.models.backbones import MobileNetV2
from mindvision.classification.models.utils import make_divisible


# Image-to-parameter module
class I2P(nn.Cell):
    """
    i2p module. The input data is x(Tensor): shape(1, 3, C, N),
    where C and N are the size of the input image.

    Returns:
        Tensor: shape(1, 62)

    Examples:
        >>> img_ori = cv2.imread("img.jpg")
        >>> transform = mindspore.dataset.vision.py_transforms.ToTensor()
        >>> img_tran = transform(img_ori)
        >>> expand_dims = ops.ExpandDims()
        >>> img = expand_dims(input_tensor, 0)
        >>> i2p = I2P()
        >>> out = i2p(x)
    """

    def __init__(self,
                 last_channel: int = 1280,
                 alpha: float = 1.0,
                 round_nearest: int = 8):
        super(I2P, self).__init__()

        self.feature_extractor = MobileNetV2()
        self.avgpool_op = ops.AdaptiveAvgPool2D(1)
        self.last_channel = make_divisible(last_channel * max(1.0, alpha), round_nearest)

        self.num_ori = 12
        self.num_shape = 40
        self.num_exp = 10

        # building classifier(orientation/shape/expression)
        self.classifier_ori = nn.SequentialCell(
            nn.Dropout(0.2),
            nn.Dense(self.last_channel, self.num_ori),
        )
        self.classifier_shape = nn.SequentialCell(
            nn.Dropout(0.2),
            nn.Dense(self.last_channel, self.num_shape),
        )
        self.classifier_exp = nn.SequentialCell(
            nn.Dropout(0.2),
            nn.Dense(self.last_channel, self.num_exp),
        )

    def construct(self, x):
        """construct."""
        x = self.feature_extractor(x)
        pool = self.avgpool_op(x)
        x = pool.reshape(x.shape[0], -1)
        x_ori = self.classifier_ori(x)
        x_shape = self.classifier_shape(x)
        x_exp = self.classifier_exp(x)
        concat_op = ops.Concat(1)
        out = concat_op((x_ori, x_shape, x_exp))

        return out
