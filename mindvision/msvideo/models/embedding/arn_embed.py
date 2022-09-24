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
""" Embendding for ARN based on Unit3d-built Conv-4 or C3d Backbone."""

from typing import Optional

from mindspore import nn
from mindspore import ops as P

from mindvision.msvideo.models.blocks.unit3d import Unit3D
from mindvision.msvideo.models.backbones.c3d import C3D


class ARNEmbedding(nn.Cell):
    """
    Embendding for ARN based on Unit3d-built 4-layer Conv or C3d.

    Args:
        support_num_per_class (int): Number of samples in support set per class. Default: 5.
        query_num_per_class (int): Number of samples in query set per class. Default: 3.
        class_num (int): Number of classes. Default: 5.
        is_c3d (bool): Specifies whether the network uses C3D as embendding for ARN. Default: False.
        in_channels: The number of channels of the input feature. Default: 3.
        out_channels: The number of channels of the output of hidden layers (only used when is_c3d is set to False).
            Default: 64.
        pooling: The definition of pooling layer (only used when is_c3d is set to False).
            Default: ops.MaxPool3D(kernel_size=2, strides=2).

    Returns:
        Tensor, output 2 tensors.
    """

    def __init__(self,
                 support_num_per_class: int = 5,
                 query_num_per_class: int = 3,
                 class_num: int = 5,
                 is_c3d: bool = False,
                 in_channels: Optional[int] = 3,
                 out_channels: Optional[int] = 64,
                 pooling: Optional[nn.Cell] = P.MaxPool3D(
                     kernel_size=2, strides=2)
                 ) -> None:
        super(ARNEmbedding, self).__init__()
        self.support_num_per_class = support_num_per_class
        self.query_num_per_class = query_num_per_class
        self.class_num = class_num

        if is_c3d:
            # reusing c3d backbone as embendding
            self.embendding = C3D(in_channels)
        else:
            # reusing unit3d block for building Conv-4 architecture as embendding
            self.embendding = nn.SequentialCell(
                Unit3D(in_channels, out_channels, pooling=pooling),
                Unit3D(out_channels, out_channels, pooling=pooling),
                Unit3D(out_channels, out_channels),
                Unit3D(out_channels, out_channels)
            )

    def construct(self, data):
        """Construct embendding for ARN."""
        support = data[0, :self.support_num_per_class *
                       self.class_num, :, :, :, :]
        query = data[0, self.support_num_per_class*self.class_num:self.support_num_per_class*self.class_num +
                     self.query_num_per_class*self.class_num, :, :, :, :]

        support_features = self.embendding(support)
        query_features = self.embendding(query)

        return support_features, query_features
