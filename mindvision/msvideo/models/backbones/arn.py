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
""" ARN backbone."""

from mindspore import nn

from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.msvideo.models.blocks.spatial_attention import SpatialAttention
from mindvision.msvideo.models.blocks.jigsaw_discriminator3d import JigsawDiscriminator3d


@ClassFactory.register(ModuleType.BACKBONE)
class ARNBackbone(nn.Cell):
    """ARN architecture. TODO: these architecture is slight complex. we will discuses later.

    Args:
        jigsaw (int): Number of the output dimension for spacial-temporal jigsaw discriminator. Default: 10.
        support_num_per_class (int): Number of samples in support set per class. Default: 5.
        query_num_per_class (int): Number of samples in query set per class. Default: 3.
        class_num (int): Number of classes. Default: 5.

    Returns:
        Tensor, output 2 tensors.

    Examples:
        >>> ARNBackbone(10, 5, 3, 5)
    """

    def __init__(self,
                 jigsaw: int = 10,
                 support_num_per_class: int = 5,
                 query_num_per_class: int = 3,
                 class_num: int = 5
                 ):
        super(ARNBackbone, self).__init__()
        self.jigsaw = jigsaw

        self.support_num_per_class = support_num_per_class
        self.query_num_per_class = query_num_per_class
        self.class_num = class_num

        self.jigsaw_discriminator = JigsawDiscriminator3d(64, self.jigsaw)

    def construct(self, support_features, query_features):
        """test construct of arn backbone"""
        channel = support_features.shape[1]
        temporal_dim = support_features.shape[2]
        width = support_features.shape[3]
        height = support_features.shape[4]

        spatial_detector = SpatialAttention(channel, 16)

        support_ta = 1 + spatial_detector(support_features)
        query_ta = 1 + spatial_detector(query_features)

        support_features = (support_features * support_ta).reshape(
            self.support_num_per_class * self.class_num,
            channel, temporal_dim * width * height)  # C * N
        query_features = (query_features * query_ta).reshape(
            self.query_num_per_class * self.class_num,
            channel, temporal_dim * width * height)  # C * N

        return support_features, query_features
