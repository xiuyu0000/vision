# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""SynergyNet utils"""
import random
import mindspore
from mindspore import ops
from mindspore import Tensor


class Crop:
    """
    Input:
        Tensor: shape(3, 120, 120),
    Return:
        Tensor: shape(3, 120, 120)
    """

    def __init__(self, maximum, std=None, prob=0.01, mode='test'):
        self.maximum = maximum
        self.std = std
        self.prob = prob
        self.type_li = [1, 2, 3, 4, 5, 6, 7]
        self.switcher = {
            1: self.lup,
            2: self.rup,
            3: self.ldown,
            4: self.rdown,
            5: self.lhalf,
            6: self.rhalf,
            7: self.center
        }
        self.mode = mode
        self.zero = ops.Zeros()

    def get_params(self, img):
        h = img.shape[1]
        w = img.shape[2]
        crop_margins = self.maximum
        rand = random.random()

        return crop_margins, h, w, rand

    def lup(self, img, h, w):
        new_img = self.zero((3, h, w), mindspore.float32)
        new_img[:, :h // 2, :w // 2] = img[:, :h // 2, :w // 2]
        return new_img

    def rup(self, img, h, w):
        new_img = self.zero((3, h, w), mindspore.float32)
        new_img[:, :h // 2, w // 2:] = img[:, :h // 2, w // 2:]
        return new_img

    def ldown(self, img, h, w):
        new_img = self.zero((3, h, w), mindspore.float32)
        new_img[:, h // 2:, :w // 2] = img[:, h // 2:, :w // 2]
        return new_img

    def rdown(self, img, h, w):
        new_img = self.zero((3, h, w), mindspore.float32)
        new_img[:, :h // 2, :w // 2] = img[:, :h // 2, :w // 2]
        return new_img

    def lhalf(self, img, h, w):
        new_img = self.zero((3, h, w), mindspore.float32)
        new_img[:, :, :w // 2] = img[:, :, :w // 2]
        return new_img

    def rhalf(self, img, h, w):
        new_img = self.zero((3, h, w), mindspore.float32)
        new_img[:, :, w // 2:] = img[:, :, w // 2:]
        return new_img

    def center(self, img, h, w):
        new_img = self.zero((3, h, w), mindspore.float32)
        new_img[:, h // 4: -h // 4, w // 4: -w // 4] = img[:, h // 4: -h // 4, w // 4: -w // 4]
        return new_img

    def __call__(self, img, gt=None):
        img_tensor = Tensor(img, dtype=mindspore.float32)
        crop_margins, h, w, rand = self.get_params(img_tensor)
        crop_backgnd = self.zero((3, h, w), mindspore.float32)

        crop_backgnd[:, crop_margins:h - 1 * crop_margins, crop_margins:w - 1 * crop_margins] = \
            img_tensor[:, crop_margins: h - crop_margins, crop_margins: w - crop_margins]
        # random center crop
        if (rand < self.prob) and (self.mode == 'train'):
            func = self.switcher.get(random.randint(1, 7))
            crop_backgnd = func(crop_backgnd, h, w)

        # center crop
        if self.mode == 'test':
            crop_backgnd[:, crop_margins:h - 1 * crop_margins, crop_margins:w - 1 * crop_margins] = \
                img_tensor[:, crop_margins: h - crop_margins, crop_margins: w - crop_margins]
            crop_backgnd = crop_backgnd.asnumpy()
        return crop_backgnd
