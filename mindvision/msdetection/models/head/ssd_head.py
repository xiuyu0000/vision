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
"""Head for SSD."""

from typing import List, Union
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from mindvision.classification.models.blocks import ConvNormActivation
from mindvision.msdetection.loss.sigmoid_focal_classification_loss import SigmoidFocalClassificationLoss
from mindvision.msdetection.internals.anchor import GenerateDefaultBoxes, GridAnchorGenerator
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["WeightSharedMultiBox", "MultiBox"]


class FlattenConcat(nn.Cell):
    """
    Concatenate predictions into a single tensor.
    """

    def __init__(self, num_ssd_boxes: int):
        super(FlattenConcat, self).__init__()
        self.num_ssd_boxes = num_ssd_boxes
        self.concat = ops.Concat(axis=1)
        self.transpose = ops.Transpose()

    def construct(self, inputs):
        output = ()
        batch_size = inputs[0].shape[0]
        for x in inputs:
            x = self.transpose(x, (0, 2, 3, 1))
            output += (x.reshape((batch_size, -1)),)
        res = self.concat(output)
        return res.reshape((batch_size, self.num_ssd_boxes, -1))


@ClassFactory.register(ModuleType.HEAD)
class WeightSharedMultiBox(nn.Cell):
    """
    Weight shared Multi-box conv layers. Each multi-box layer contains class conf scores and localization predictions.
    All box predictors shares the same conv weight in different features.
    """

    def __init__(self,
                 num_classes: int,
                 extras_out_channels: List[int],
                 num_default: List[int],
                 feature_size: List[int],
                 num_addition_layers: int,
                 num_ssd_boxes: int,
                 gamma: float,
                 alpha: float,
                 anchor_generator: Union[GridAnchorGenerator, GenerateDefaultBoxes],
                 prior_scaling: List[float],
                 loc_cls_shared_addition: bool = False):
        super(WeightSharedMultiBox, self).__init__()

        out_channels = extras_out_channels[0]
        num_default = num_default[0]
        num_features = len(feature_size)

        if not loc_cls_shared_addition:
            addition_loc_layer_list = []
            addition_cls_layer_list = []
            for _ in range(num_features):
                addition_loc_layer = [
                    ConvNormActivation(out_channels, out_channels, 3, 1, activation=nn.ReLU6) for _ in
                    range(num_addition_layers)
                ]
                addition_cls_layer = [
                    ConvNormActivation(out_channels, out_channels, 3, 1, activation=nn.ReLU6) for _ in
                    range(num_addition_layers)
                ]
                addition_loc_layer_list.append(nn.SequentialCell(addition_loc_layer))
                addition_cls_layer_list.append(nn.SequentialCell(addition_cls_layer))
            self.addition_layer_loc = nn.CellList(addition_loc_layer_list)
            self.addition_layer_cls = nn.CellList(addition_cls_layer_list)
        else:
            addition_layer_list = []
            for _ in range(num_features):
                addition_layers = [
                    ConvNormActivation(out_channels, out_channels, 3, 1, activation=nn.ReLU6) for _ in
                    range(num_addition_layers)
                ]
                addition_layer_list.append(nn.SequentialCell(addition_layers))
            self.addition_layer = nn.CellList(addition_layer_list)

        loc_layers = [nn.Conv2d(in_channels=out_channels,
                                out_channels=4 * num_default,
                                kernel_size=3,
                                stride=1,
                                pad_mod='same',
                                has_bias=True)]

        cls_layers = [nn.Conv2d(in_channels=out_channels,
                                out_channels=num_classes * num_default,
                                kernel_size=3,
                                stride=1,
                                pad_mod='same',
                                has_bias=True)]

        self.loc_cls_shared_addition = loc_cls_shared_addition
        self.loc_layers = nn.SequentialCell(loc_layers)
        self.cls_layers = nn.SequentialCell(cls_layers)
        self.flatten_concat = FlattenConcat(num_ssd_boxes)
        self.less = ops.Less()
        self.tile = ops.Tile()
        self.reduce_sum = ops.ReduceSum()
        self.expand_dims = ops.ExpandDims()
        self.class_loss = SigmoidFocalClassificationLoss(gamma, alpha)
        self.loc_loss = nn.SmoothL1Loss()
        self.activation = ops.Sigmoid()
        self.default_boxes = Tensor(anchor_generator.default_boxes)
        self.prior_scaling_xy = prior_scaling[0]
        self.prior_scaling_wh = prior_scaling[1]
        self.exp = ops.Exp()
        self.concat = ops.Concat(-1)
        self.maximum = ops.Maximum()
        self.minimum = ops.Minimum()

    def construct(self, inputs):
        """forward pass"""
        loc_outputs = ()
        cls_outputs = ()
        num_heads = len(inputs)

        for i in range(num_heads):
            if self.loc_cls_shared_addition:
                features = self.addition_layer[i](inputs[i])
                loc_outputs += (self.loc_layers(features),)
                cls_outputs += (self.cls_layers(features),)
            else:
                features = self.addition_layer_loc[i](inputs[i])
                loc_outputs += (self.loc_layers(features),)
                features = self.addition_layer_cls[i](inputs[i])
                cls_outputs += (self.cls_layers(features),)

        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)

    def construct_train(self, features, boxes, labels):
        """training"""
        pred_loc, pred_label = self.construct(features)
        pred_loc = ops.cast(pred_loc, ms.float32)
        pred_label = ops.cast(pred_label, ms.float32)
        gt_loc = boxes
        gt_label = labels

        num_matched_boxes = []
        for label in labels:
            num_matched_boxes.append(len(np.nonzero(label)[0]))

        mask = ops.cast(self.less(0, gt_label), ms.float32)
        num_matched_boxes = self.reduce_sum(ops.cast(Tensor(num_matched_boxes), ms.float32))
        # Localization Loss
        mask_loc = self.tile(self.expand_dims(mask, -1), (1, 1, 4))
        smooth_l1 = self.loc_loss(pred_loc, gt_loc) * mask_loc
        loss_loc = self.reduce_sum(self.reduce_sum(smooth_l1, -1), -1)

        # Classification Loss
        loss_cls = self.class_loss(pred_label, gt_label)
        loss_cls = self.reduce_sum(loss_cls, (1, 2))
        return self.reduce_sum((loss_cls + loss_loc) / num_matched_boxes)

    def construct_eval(self, features):
        """For eval"""
        pred_loc, pred_label = self.construct(features)
        pred_label = self.activation(pred_label)
        pred_loc = ops.cast(pred_loc, ms.float32)
        pred_label = ops.cast(pred_label, ms.float32)

        default_bbox_xy = self.default_boxes[..., :2]
        default_bbox_wh = self.default_boxes[..., 2:]
        pred_xy = pred_loc[..., :2] * self.prior_scaling_xy * default_bbox_wh + default_bbox_xy
        pred_wh = self.exp(pred_loc[..., 2:] * self.prior_scaling_wh) * default_bbox_wh

        pred_xy_0 = pred_xy - pred_wh / 2.0
        pred_xy_1 = pred_xy + pred_wh / 2.0
        pred_xy = self.concat((pred_xy_0, pred_xy_1))
        pred_xy = self.maximum(pred_xy, 0)
        pred_xy = self.minimum(pred_xy, 1)
        return pred_xy, pred_label


@ClassFactory.register(ModuleType.HEAD)
class MultiBox(nn.Cell):
    """
    Multi-box conv layers. Each multi-box layer contains class conf scores and localization predictions.
    """

    def __init__(self,
                 num_classes: int,
                 extras_out_channels: List[int],
                 num_default: List[int],
                 num_ssd_boxes: int,
                 gamma: float,
                 alpha: float,
                 anchor_generator: Union[GridAnchorGenerator, GenerateDefaultBoxes],
                 prior_scaling: List[float]
                 ):
        super(MultiBox, self).__init__()

        loc_layers = []
        cls_layers = []
        for k, out_channel in enumerate(extras_out_channels):
            loc_layers += [
                nn.SequentialCell([ConvNormActivation(
                    out_channel,
                    out_channel,
                    kernel_size=3,
                    stride=1,
                    groups=out_channel,
                    activation=nn.ReLU6
                ), nn.Conv2d(out_channel, 4 * num_default[k], kernel_size=1, has_bias=True)])
            ]
            cls_layers += [
                nn.SequentialCell([ConvNormActivation(
                    out_channel,
                    out_channel,
                    kernel_size=3,
                    stride=1,
                    groups=out_channel,
                    activation=nn.ReLU6
                ), nn.Conv2d(out_channel, num_classes * num_default[k], kernel_size=1, has_bias=True)])
            ]

        self.multi_loc_layers = nn.CellList(loc_layers)
        self.multi_cls_layers = nn.CellList(cls_layers)
        self.flatten_concat = FlattenConcat(num_ssd_boxes)

        self.less = ops.Less()
        self.tile = ops.Tile()
        self.reduce_sum = ops.ReduceSum()
        self.expand_dims = ops.ExpandDims()
        self.class_loss = SigmoidFocalClassificationLoss(gamma, alpha)
        self.loc_loss = nn.SmoothL1Loss()
        self.activation = ops.Sigmoid()
        self.default_boxes = Tensor(anchor_generator.default_boxes)
        self.prior_scaling_xy = prior_scaling[0]
        self.prior_scaling_wh = prior_scaling[1]
        self.exp = ops.Exp()
        self.concat = ops.Concat(-1)
        self.maximum = ops.Maximum()
        self.minimum = ops.Minimum()

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        for i in range(len(self.multi_loc_layers)):
            loc_outputs += (self.multi_loc_layers[i](inputs[i]),)
            cls_outputs += (self.multi_cls_layers[i](inputs[i]),)
        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)

    def construct_train(self, features, boxes, labels):
        """Training."""
        pred_loc, pred_label = self.construct(features)
        pred_loc = ops.cast(pred_loc, ms.float32)
        pred_label = ops.cast(pred_label, ms.float32)
        gt_loc = boxes
        gt_label = labels

        num_matched_boxes = []
        for label in labels:
            num_matched_boxes.append(ops.count_nonzero(label))

        mask = ops.cast(self.less(0, gt_label), ms.float32)
        num_matched_boxes = self.reduce_sum(ops.cast(Tensor(num_matched_boxes), ms.float32))
        # Localization Loss
        mask_loc = self.tile(self.expand_dims(mask, -1), (1, 1, 4))
        smooth_l1 = self.loc_loss(pred_loc, gt_loc) * mask_loc
        loss_loc = self.reduce_sum(self.reduce_sum(smooth_l1, -1), -1)

        # Classification Loss
        loss_cls = self.class_loss(pred_label, gt_label)
        loss_cls = self.reduce_sum(loss_cls, (1, 2))
        return self.reduce_sum((loss_cls + loss_loc) / num_matched_boxes)

    def construct_eval(self, features):
        """Eval."""
        pred_loc, pred_label = self.construct(features)
        pred_label = self.activation(pred_label)
        pred_loc = ops.cast(pred_loc, ms.float32)
        pred_label = ops.cast(pred_label, ms.float32)

        default_bbox_xy = self.default_boxes[..., :2]
        default_bbox_wh = self.default_boxes[..., 2:]
        pred_xy = pred_loc[..., :2] * self.prior_scaling_xy * default_bbox_wh + default_bbox_xy
        pred_wh = self.exp(pred_loc[..., 2:] * self.prior_scaling_wh) * default_bbox_wh

        pred_xy_0 = pred_xy - pred_wh / 2.0
        pred_xy_1 = pred_xy + pred_wh / 2.0
        pred_xy = self.concat((pred_xy_0, pred_xy_1))
        pred_xy = self.maximum(pred_xy, 0)
        pred_xy = self.minimum(pred_xy, 1)
        return pred_xy, pred_label
