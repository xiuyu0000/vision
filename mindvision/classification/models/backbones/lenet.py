# Copyright 2021
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
"""LeNet5 backbone."""

from mindspore import nn
from mindspore.common.initializer import Normal

from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["LeNet5"]


@ClassFactory.register(ModuleType.BACKBONE)
class LeNet5(nn.Cell):
    """
    LeNet backbone.

    Args:
        num_class (int): The number of classes. Default: 10.
        num_channel (int): The number of channels. Default: 1.
        include_top (bool): Whether to use the TOP architecture. Default: True.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`

    Outputs:
        Tensor of shape :math:`(N, 10)`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models.backbones import LeNet5
        >>>
        >>> net = LeNet5()
        >>> x = ms.Tensor(np.ones([1, 1, 32, 32]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 10)

    About LeNet5:

    LeNet5 trained with the back-propagation algorithm constitute the best example of
    a successful gradient based learning technique. Given an appropriate network architecture,
    gradient-based learning algorithms can be used to synthesize a complex decision surface
    that can classify high-dimensional patterns, such as handwritten characters, with
    minimal preprocessing.

    Citation:

    .. code-block::

        @article{1998Gradient,
          title={Gradient-based learning applied to document recognition},
          author={ Lecun, Y.  and  Bottou, L. },
          journal={Proceedings of the IEEE},
          volume={86},
          number={11},
          pages={2278-2324},
          year={1998}
        }
    """

    def __init__(self, num_classes=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.include_top = include_top

        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_classes, weight_init=Normal(0.02))

    def construct(self, x):
        """
        LeNet5 construct.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if self.include_top:
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
        return x
