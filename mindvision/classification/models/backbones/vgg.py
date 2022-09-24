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
"""VGG backbone."""

from typing import List, Dict, Union

from mindspore import nn

from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = [
    "VGG",
    "VGG11",
    "VGG13",
    "VGG16",
    "VGG19"
]

cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layer(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.SequentialCell:
    """Make stage network of VGG."""

    layers = []
    in_channels = 3

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=1,
                               pad_mode="pad",
                               has_bias=True)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]

            in_channels = v  # pylint=unused-argument

    return nn.SequentialCell(layers)


class VGG(nn.Cell):
    """VGG backbone."""

    def __init__(self, model_name: str, batch_norm: bool = False):
        super(VGG, self).__init__()
        cfg = cfgs[model_name]
        self.features = _make_layer(cfg, batch_norm)
        self.flatten = nn.Flatten()

    def construct(self, x):
        """
        Vgg construct.
        """
        x = self.features(x)
        x = self.flatten(x)

        return x


@ClassFactory.register(ModuleType.BACKBONE)
class VGG11(VGG):
    """
    The class of VGG11 uses the registration mechanism to register, need to use the yaml configuration file to call.
    """

    def __init__(self, batch_norm: bool = False):
        super(VGG11, self).__init__(model_name="vgg11", batch_norm=batch_norm)


@ClassFactory.register(ModuleType.BACKBONE)
class VGG13(VGG):
    """
    The class of VGG13 uses the registration mechanism to register, need to use the yaml configuration file to call.
    """

    def __init__(self, batch_norm: bool = False):
        super(VGG13, self).__init__(model_name="vgg13", batch_norm=batch_norm)


@ClassFactory.register(ModuleType.BACKBONE)
class VGG16(VGG):
    """
    The class of VGG16 uses the registration mechanism to register, need to use the yaml configuration file to call.
    """

    def __init__(self, batch_norm: bool = False):
        super(VGG16, self).__init__(model_name="vgg16", batch_norm=batch_norm)


@ClassFactory.register(ModuleType.BACKBONE)
class VGG19(VGG):
    """
    The class of VGG19 uses the registration mechanism to register, need to use the yaml configuration file to call.
    """

    def __init__(self, batch_norm: bool = False):
        super(VGG19, self).__init__(model_name="vgg19", batch_norm=batch_norm)
