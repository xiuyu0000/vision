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
"""Inception3d Module."""

from mindspore import ops
from mindspore import nn
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.msvideo.models.blocks.unit3d import Unit3D


@ClassFactory.register(ModuleType.BACKBONE)
class Inception3dModule(nn.Cell):
    """
    Inception3dModule definition.

    Args:
        in_channels (int):  The number of channels of input frame images.
        out_channels (int): The number of channels of output frame images.

    Returns:
        Tensor, output tensor.

    Examples:
        Inception3dModule(in_channels=3, out_channels=3)
    """

    def __init__(self, in_channels, out_channels):
        super(Inception3dModule, self).__init__()
        self.cat = ops.Concat(axis=1)
        self.b0 = Unit3D(
            in_channels=in_channels,
            out_channels=out_channels[0],
            kernel_size=(1, 1, 1))
        self.b1a = Unit3D(
            in_channels=in_channels,
            out_channels=out_channels[1],
            kernel_size=(1, 1, 1))
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            out_channels=out_channels[2],
            kernel_size=(3, 3, 3))
        self.b2a = Unit3D(
            in_channels=in_channels,
            out_channels=out_channels[3],
            kernel_size=(1, 1, 1))
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            out_channels=out_channels[4],
            kernel_size=(3, 3, 3))
        self.b3a = ops.MaxPool3D(
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            pad_mode="same")
        self.b3b = Unit3D(
            in_channels=in_channels,
            out_channels=out_channels[5],
            kernel_size=(1, 1, 1))

    def construct(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return self.cat((b0, b1, b2, b3))
