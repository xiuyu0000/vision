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
"""ARN head."""

from mindspore import nn
from mindspore import ops

from mindvision.classification.models.blocks import ConvNormActivation
from mindvision.engine.class_factory import ClassFactory, ModuleType


class SimilarityNetwork(nn.Cell):
    """Similarity learning between query and support clips as paired
    relation descriptors for RelationNetwork.

    Args:
        in_channels (int): Number of channels of the input feature. Default: 2.
        out_channels (int): Number of channels of the output feature. Default: 64.
        input_size (int): Size of input features. Default: 64.
        hidden_size (int): Number of channels in the hidden fc layers. Default: 8.
    Returns:
        Tensor, output tensor.
    """

    def __init__(self, in_channels=2, out_channels=64, input_size=64, hidden_size=8):
        super(SimilarityNetwork, self).__init__()

        self.layer1 = ConvNormActivation(in_channels, out_channels)
        self.layer2 = ConvNormActivation(out_channels, out_channels)
        self.layer3 = ConvNormActivation(out_channels, out_channels)
        self.layer4 = ConvNormActivation(out_channels, out_channels)

        self.fc1 = nn.Dense(out_channels * (input_size // 2 // 2 // 2 // 2)
                            * (input_size // 2 // 2 // 2 // 2), hidden_size)
        self.fc2 = nn.Dense(hidden_size, 1)
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        out = self.maxpool2d(self.layer1(x))
        out = self.maxpool2d(out + self.layer2(out))
        out = self.maxpool2d(out + self.layer3(out))
        out = self.maxpool2d(out + self.layer4(out))
        out = out.reshape(out.shape[0], -1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out


@ClassFactory.register(ModuleType.HEAD)
class ARNHead(nn.Cell):
    """
    ARN head architecture.

    Args:
        class_num (int): Number of classes. Default: 5.
        query_num_per_class (int): Number of query samples per class. Default: 3.

    Returns:
        Tensor, output tensor.
    """

    def __init__(self,
                 class_num: int = 5,
                 query_num_per_class: int = 3
                 ):
        super(ARNHead, self).__init__()
        self.class_num = class_num
        self.query_num_per_class = query_num_per_class

        self.expand = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.cat_relation = ops.Concat(axis=2)

    def construct(self, so_support_features, so_query_features):
        """test construct of arn head"""
        channel = so_support_features.shape[-1]
        relation_network = SimilarityNetwork(
            out_channels=channel, input_size=channel)

        support_feature_ex = self.expand(so_support_features, 0).repeat(self.query_num_per_class * self.class_num,
                                                                        axis=0)

        query_feature_ex = self.expand(
            so_query_features, 0).repeat(self.class_num, axis=0)
        query_feature_ex = self.transpose(query_feature_ex, (1, 0, 2, 3, 4))

        relation_pairs = self.cat_relation(
            (support_feature_ex, query_feature_ex)).reshape(-1, 2, channel, channel)
        relations = relation_network(
            relation_pairs).reshape(-1, self.class_num)  # query_num * class_num

        return relations
