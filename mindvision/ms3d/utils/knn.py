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
"""KNN module"""
from mindspore import ops


class KNN:
    """
        KNN Module.

        The input data is x shape(B,C,N)
        where B is the batch size ,C is the dimension of the transform matrix
        and N is the number of points.

        :param x: input data
        :param k: k-NearestNeighbor Parameter
        :return: Tensor shape(B,N,K)
    """

    def __init__(self, x, k):
        self.x = x
        self.k = k

    def func(self):
        """KNN func"""
        inner = -2 * ops.matmul(self.x.transpose(0, 2, 1), self.x)
        xx = -2 * (self.x ** 2).sum(axis=1, keepdims=True)
        pairwise_distance = -xx - inner - xx.transpose(0, 2, 1)
        idx, _ = ops.TopK(sorted=True)(pairwise_distance, self.k)
        return idx
