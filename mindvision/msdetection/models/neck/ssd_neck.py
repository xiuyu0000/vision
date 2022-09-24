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
"""Neck for SSD."""

from typing import List

import mindspore.nn as nn
import mindspore.ops as ops

from mindvision.classification.models.backbones.mobilenet_v2 import InvertedResidual
from mindvision.classification.models.blocks import ConvNormActivation
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["SSDMobileNetV1FPN", "SSDResNet50FPN", "SSDMobileNetV2Neck"]


class FpnTopDown(nn.Cell):
    """
    Fpn to extract features
    """

    def __init__(self, in_channel_list, out_channels):
        super(FpnTopDown, self).__init__()
        self.lateral_convs_list_ = []
        self.fpn_convs_ = []
        for channel in in_channel_list:
            l_conv = nn.Conv2d(channel, out_channels, kernel_size=1, stride=1,
                               has_bias=True, padding=0, pad_mode='same')
            fpn_conv = ConvNormActivation(out_channels, out_channels, activation=nn.ReLU6)
            self.lateral_convs_list_.append(l_conv)
            self.fpn_convs_.append(fpn_conv)
        self.lateral_convs_list = nn.CellList(self.lateral_convs_list_)
        self.fpn_convs_list = nn.CellList(self.fpn_convs_)
        self.num_layers = len(in_channel_list)

    def construct(self, inputs):
        """forward pass"""
        image_features = ()
        for i, feature in enumerate(inputs):
            image_features = image_features + (self.lateral_convs_list[i](feature),)

        features = (image_features[-1],)
        for i in range(len(inputs) - 1):
            top = len(inputs) - i - 1
            down = top - 1
            size = ops.shape(inputs[down])
            top_down = ops.ResizeBilinear((size[2], size[3]))(features[-1])
            top_down = top_down + image_features[down]
            features = features + (top_down,)

        extract_features = ()
        num_features = len(features)
        for i in range(num_features):
            extract_features = extract_features + (self.fpn_convs_list[i](features[num_features - i - 1]),)

        return extract_features


class BottomUp(nn.Cell):
    """
    Bottom Up feature extractor
    """

    def __init__(self, levels, channels, kernel_size, stride):
        super(BottomUp, self).__init__()
        self.levels = levels
        bottom_up_cells = [
            ConvNormActivation(channels, channels, kernel_size, stride) for _ in range(self.levels)
        ]
        self.blocks = nn.CellList(bottom_up_cells)

    def construct(self, features):
        for block in self.blocks:
            features = features + (block(features[-1]),)
        return features


class FeatureSelector(nn.Cell):
    """
    Select specific layers from an entire feature list
    """

    def __init__(self, feature_idxes):
        super(FeatureSelector, self).__init__()
        self.feature_idxes = feature_idxes

    def construct(self, feature_list):
        selected = ()
        for i in self.feature_idxes:
            selected = selected + (feature_list[i],)
        return selected


@ClassFactory.register(ModuleType.NECK)
class SSDMobileNetV1FPN(nn.Cell):
    """
    MobileNetV1 with FPN as Ssd neck.
    """

    def __init__(self):
        super(SSDMobileNetV1FPN, self).__init__()
        self.selector = FeatureSelector([10, 22, 26])
        self.fpn = FpnTopDown([256, 512, 1024], 256)
        self.bottom_up = BottomUp(2, 256, 3, 2)

    def construct(self, x):
        x = self.selector(x)
        x = self.fpn(x)
        x = self.bottom_up(x)
        return x


@ClassFactory.register(ModuleType.NECK)
class SSDResNet50FPN(nn.Cell):
    """
    ResNet with FPN as SSD backbone.
    """

    def __init__(self):
        super(SSDResNet50FPN, self).__init__()
        self.fpn = FpnTopDown([512, 1024, 2048], 256)
        self.bottom_up = BottomUp(2, 256, 3, 2)

    def construct(self, x):
        features = self.fpn(x)
        features = self.bottom_up(features)
        return features


@ClassFactory.register(ModuleType.NECK)
class SSDMobileNetV2Neck(nn.Cell):
    """
    MobileNetV2 as SSD backbone.
    """
    def __init__(self,
                 extras_in_channels: List[int],
                 extras_out_channels: List[int],
                 extras_strides: List[int],
                 extras_ratios: List[float]):
        super(SSDMobileNetV2Neck, self).__init__()
        residual_list = []
        for i in range(2, len(extras_in_channels)):
            residual = InvertedResidual(extras_in_channels[i],
                                        extras_out_channels[i],
                                        stride=extras_strides[i],
                                        expand_ratio=extras_ratios[i],
                                        last_relu=True
                                        )
            residual_list.append(residual)

        self.multi_residual = nn.CellList(residual_list)

    def construct(self, x):
        expand_layer, output = x
        multi_feature = (expand_layer, output)
        feature = output

        for residual in self.multi_residual:
            feature = residual(feature)
            multi_feature += (feature,)

        return multi_feature
