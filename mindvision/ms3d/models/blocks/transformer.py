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
"""Transformer Attention Module."""

from mindspore import ops
import mindspore.numpy as mnp
from mindspore import nn
from pointnet2_utils import index_points, square_distance


class Transformer(nn.Cell):
    """
        Transformer Block
        Input:
            features(Tensor): set of feature vectors.
            xyz(Tensor): vectors associated 3D coordinates.
        Return:
            New feature vectors for all data points.
        """

    def __init__(self, d_points, d_model, k):
        super(Transformer, self).__init__()
        self.fc1 = nn.Dense(d_points, d_model)
        self.fc2 = nn.Dense(d_model, d_points)
        self.fc_delta = nn.SequentialCell(
            nn.Dense(3, d_model),
            ops.ReLU(),
            nn.Dense(d_model, d_model)
        )
        self.fc_gamma = nn.SequentialCell(
            nn.Dense(d_model, d_model),
            ops.ReLU(),
            nn.Dense(d_model, d_model)
        )
        self.w_qs = nn.Dense(d_model, d_model, has_bias=False)
        self.w_ks = nn.Dense(d_model, d_model, has_bias=False)
        self.w_vs = nn.Dense(d_model, d_model, has_bias=False)
        self.k = k

    def construct(self, xyz, features):
        """
        Transformer construct
        """
        dist = square_distance(xyz, xyz)
        knn_idx = dist.argsort()[:, :, :self.k]
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = nn.Softmax(attn / mnp.sqrt(k.size(-1)), axis=-2)  # b x n x k x f
        einsum = ops.Einsum("bmnf, bmnf->bmf")
        res = einsum((attn, v + pos_enc))
        res = self.fc2(res) + pre
        return res, attn
