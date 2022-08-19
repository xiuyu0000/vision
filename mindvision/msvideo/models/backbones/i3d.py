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
"""I3D backbone."""

from mindspore import nn
from mindspore import ops
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.msvideo.models.builder import build_backbone
from mindvision.msvideo.models.blocks.unit3d import Unit3D


@ClassFactory.register(ModuleType.BACKBONE)
class InceptionI3d(nn.Cell):
    """
    InceptionI3d architecture. TODO: i3d Inception backbone just in 3d?what about 2d. and two steam.

    Args:
        in_channels (int): The number of channels of input frame images(default 3).
    Returns:
        Tensor, output tensor.

    Examples:
        >>> InceptionI3d(in_channels=3)
    """

    def __init__(self, in_channels=3):

        super(InceptionI3d, self).__init__()

        self.conv3d_1a_7x7 = Unit3D(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2))
        self.maxpool3d_2a_3x3 = ops.MaxPool3D(
            kernel_size=(1, 3, 3),
            strides=(1, 2, 2),
            pad_mode="same")

        self.conv3d_2b_1x1 = Unit3D(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1, 1))

        self.conv3d_2c_3x3 = Unit3D(
            in_channels=64,
            out_channels=192,
            kernel_size=(3, 3, 3))

        self.maxpool3d_3a_3x3 = ops.MaxPool3D(
            kernel_size=(1, 3, 3),
            strides=(1, 2, 2),
            pad_mode="same")

        self.mixed_3b = build_backbone(
            {
                "type": "Inception3dModule",
                "in_channels": 192,
                "out_channels": [64, 96, 128, 16, 32, 32]})

        self.mixed_3c = build_backbone(
            {
                "type": "Inception3dModule",
                "in_channels": 256,
                "out_channels": [128, 128, 192, 32, 96, 64]})

        self.maxpool3d_4a_3x3 = ops.MaxPool3D(
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            pad_mode="same")

        self.mixed_4b = build_backbone(
            {
                "type": "Inception3dModule",
                "in_channels": 128 + 192 + 96 + 64,
                "out_channels": [192, 96, 208, 16, 48, 64]})

        self.mixed_4c = build_backbone(
            {
                "type": "Inception3dModule",
                "in_channels": 192 + 208 + 48 + 64,
                "out_channels": [160, 112, 224, 24, 64, 64]})

        self.mixed_4d = build_backbone(
            {
                "type": "Inception3dModule",
                "in_channels": 160 + 224 + 64 + 64,
                "out_channels": [128, 128, 256, 24, 64, 64]})

        self.mixed_4e = build_backbone(
            {
                "type": "Inception3dModule",
                "in_channels": 128 + 256 + 64 + 64,
                "out_channels": [112, 144, 288, 32, 64, 64]})

        self.mixed_4f = build_backbone(
            {
                "type": "Inception3dModule",
                "in_channels": 112 + 288 + 64 + 64,
                "out_channels": [256, 160, 320, 32, 128, 128]})

        self.maxpool3d_5a_2x2 = ops.MaxPool3D(
            kernel_size=(2, 2, 2),
            strides=(2, 2, 2),
            pad_mode="same")

        self.mixed_5b = build_backbone(
            {
                "type": "Inception3dModule",
                "in_channels": 256 + 320 + 128 + 128,
                "out_channels": [256, 160, 320, 32, 128, 128]})

        self.mixed_5c = build_backbone(
            {
                "type": "Inception3dModule",
                "in_channels": 256 + 320 + 128 + 128,
                "out_channels": [384, 192, 384, 48, 128, 128]})

        self.mean_op = ops.ReduceMean(keep_dims=True)
        self.concat_op = ops.Concat(axis=2)
        self.stridedslice_op = ops.StridedSlice()

    def construct(self, x):
        """Average pooling 3D construct."""
        x = self.conv3d_1a_7x7(x)
        x = self.maxpool3d_2a_3x3(x)
        x = self.conv3d_2b_1x1(x)
        x = self.conv3d_2c_3x3(x)
        x = self.maxpool3d_3a_3x3(x)
        x = self.mixed_3b(x)
        x = self.mixed_3c(x)
        x = self.maxpool3d_4a_3x3(x)
        x = self.mixed_4b(x)
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        x = self.mixed_4f(x)
        x = self.maxpool3d_5a_2x2(x)
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        return x
