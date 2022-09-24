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
# ==============================================================================
"""S3FD neck"""

from mindspore import nn
from mindspore import ops
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.msdetection.engine.ops import L2Norm

__all__ = ["S3FDNeck"]


@ClassFactory.register(ModuleType.NECK)
class S3FDNeck(nn.Cell):
    """S3FD neck implementation."""

    def __init__(self):
        super(S3FDNeck, self).__init__()

        self.conv3_3_norm = L2Norm(256, scale=10)
        self.conv4_3_norm = L2Norm(512, scale=8)
        self.conv5_3_norm = L2Norm(512, scale=5)

        self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, kernel_size=3, stride=1,
                                                pad_mode="pad", padding=1, has_bias=True)
        self.conv3_3_norm_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1,
                                               pad_mode="pad", padding=1, has_bias=True)
        self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1,
                                                pad_mode="pad", padding=1, has_bias=True)
        self.conv4_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1,
                                               pad_mode="pad", padding=1, has_bias=True)
        self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1,
                                                pad_mode="pad", padding=1, has_bias=True)
        self.conv5_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1,
                                               pad_mode="pad", padding=1, has_bias=True)

        self.fc7_mbox_conf = nn.Conv2d(1024, 2, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True)
        self.fc7_mbox_loc = nn.Conv2d(1024, 4, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True)
        self.conv6_2_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True)
        self.conv6_2_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True)
        self.conv7_2_mbox_conf = nn.Conv2d(256, 2, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True)
        self.conv7_2_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True)

    def construct(self, x):
        """S3FD neck construct."""

        f3_3 = self.conv3_3_norm(x[0])
        f4_3 = self.conv4_3_norm(x[1])
        f5_3 = self.conv5_3_norm(x[2])

        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)
        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        cls4 = self.fc7_mbox_conf(x[3])
        reg4 = self.fc7_mbox_loc(x[3])
        cls5 = self.conv6_2_mbox_conf(x[4])
        reg5 = self.conv6_2_mbox_loc(x[4])
        cls6 = self.conv7_2_mbox_conf(x[5])
        reg6 = self.conv7_2_mbox_loc(x[5])

        # max-out background label
        chunk = ops.Split(1, 4)(cls1)
        bmax = ops.Maximum()(ops.Maximum()(chunk[0], chunk[1]), chunk[2])
        cls1 = ops.Concat(1)((bmax, chunk[3]))

        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]
