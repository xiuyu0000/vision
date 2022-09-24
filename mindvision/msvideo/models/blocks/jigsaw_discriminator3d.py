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
""" Spacial-Temporal Jigsaw Discriminator(3D) Module."""

from mindspore import nn

from mindvision.msvideo.engine.ops import MaxPool3D
from mindvision.msvideo.models.blocks.unit3d import Unit3D


class JigsawDiscriminator3d(nn.Cell):
    """
    Spacial-temporal jigsaw discriminator block definition.

    Args:
        in_dim (int):  The number of the input dimension for spacial-temporal jigsaw discriminator.
        out_dim (int):  The number of the output dimension for spacial-temporal jigsaw discriminator.

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, in_dim, out_dim):
        super(JigsawDiscriminator3d, self).__init__()
        self.layer1 = Unit3D(in_channels=in_dim, out_channels=64)
        self.layer2 = Unit3D(in_channels=64, out_channels=64)
        self.maxpool3d = MaxPool3D(kernel_size=2, strides=2)
        self.fc1 = nn.Dense(4096, 1024)
        self.fc2 = nn.Dense(1024, out_dim)
        self.relu = nn.ReLU()

    def construct(self, x):
        """construct of spacial-temporal jigsaw discriminator block"""
        out = self.maxpool3d(self.layer1(x))  # 64 x 2 x 16 x 16
        out = self.maxpool3d(self.layer2(out))  # 64 x 1 x 8 x 8
        out = out.view(out.shape[0], -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
