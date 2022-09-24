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
"""TransitionDown Module"""

from mindvision.ms3d.models.blocks import pointnet2_sa
from mindspore import nn


class TransitionDown(nn.Cell):
    """
    TransitionDown Block

    Input:
        xyz: input points position data, [B, C, N]
        points: input points data, [B, D, N]

    Return:
        new_xyz: sampled points position data, [B, C, S]
        new_points_concat: sample points feature data, [B, D', S]
    """

    def __init__(self, k, nneighbor, channels):
        super(Transitiondown, self).__init__()
        self.sa = pointnet2_sa.PointNet2SetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False)

    def construct(self, xyz, points):
        """
        TransitionDown construct
        """
        after_sa = self.sa(xyz, points)
        return after_sa
