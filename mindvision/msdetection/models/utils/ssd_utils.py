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
"""SSD utils."""

from typing import List, Union
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size

from mindvision.msdetection.internals.anchor import GridAnchorGenerator, GenerateDefaultBoxes
from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.ENCODER)
class SSDEncoder:
    """
    SSD encoder.
    """

    def __init__(self,
                 match_threshold: float,
                 prior_scaling: List[float],
                 anchor_generator: Union[GridAnchorGenerator, GenerateDefaultBoxes]
                 ):

        self.prior_scaling = prior_scaling
        self.matching_threshold = match_threshold
        self.default_boxes_tlbr = anchor_generator.default_boxes_tlbr
        self.default_boxes = anchor_generator.default_boxes
        self.num_ssd_boxes = len(self.default_boxes)
        self.x1, self.y1, self.x2, self.y2 = np.split(self.default_boxes_tlbr[:, :4], 4, axis=-1)
        self.vol_anchors = (self.x2 - self.x1) * (self.y2 - self.y1)

    def bboxes_encode(self, boxes, labels):
        """Labels anchors with ground truth inputs."""

        def jaccard_with_anchors(bbox):
            """Compute jaccard score a box and the anchors."""
            # Intersection bbox and volume.
            xmin = np.maximum(self.x1, bbox[0])
            ymin = np.maximum(self.y1, bbox[1])
            xmax = np.minimum(self.x2, bbox[2])
            ymax = np.minimum(self.y2, bbox[3])
            w = np.maximum(xmax - xmin, 0.)
            h = np.maximum(ymax - ymin, 0.)

            # Volumes.
            inter_vol = h * w
            union_vol = self.vol_anchors + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter_vol
            jaccard = inter_vol / union_vol
            return np.squeeze(jaccard)

        pre_scores = np.zeros(self.num_ssd_boxes, dtype=np.float32)
        t_boxes = np.zeros((self.num_ssd_boxes, 4), dtype=np.float32)
        t_label = np.zeros(self.num_ssd_boxes, dtype=np.int64)

        for bbox, label in zip(boxes, labels):
            label = int(label)
            scores = jaccard_with_anchors(bbox)
            idx = np.argmax(scores)
            scores[idx] = 2.0
            mask = (scores > self.matching_threshold)
            mask = mask & (scores > pre_scores)
            pre_scores = np.maximum(pre_scores, scores * mask)
            t_label = mask * label + (1 - mask) * t_label
            for i in range(4):
                t_boxes[:, i] = mask * bbox[i] + (1 - mask) * t_boxes[:, i]

        index = np.nonzero(t_label)

        bboxes = np.zeros((self.num_ssd_boxes, 4), dtype=np.float32)
        bboxes[:, [0, 1]] = (t_boxes[:, [0, 1]] + t_boxes[:, [2, 3]]) / 2
        bboxes[:, [2, 3]] = t_boxes[:, [2, 3]] - t_boxes[:, [0, 1]]

        # Encode features.
        bboxes_t = bboxes[index]
        default_boxes_t = self.default_boxes[index]
        bboxes_t[:, :2] = (bboxes_t[:, :2] - default_boxes_t[:, :2]) / (default_boxes_t[:, 2:] * self.prior_scaling[0])
        tmp = np.maximum(bboxes_t[:, 2:4] / default_boxes_t[:, 2:4], 0.000001)
        bboxes_t[:, 2:4] = np.log(tmp) / self.prior_scaling[1]
        bboxes[index] = bboxes_t

        return bboxes, t_label.astype(np.int32)

    def bboxes_decode(self, boxes: np.ndarray):
        """Decode predict boxes to [y, x, h, w]"""
        boxes_t = boxes.copy()
        default_boxes_t = self.default_boxes.copy()
        boxes_t[:, :2] = boxes_t[:, :2] * self.prior_scaling[0] * default_boxes_t[:, 2:] + default_boxes_t[:, :2]
        boxes_t[:, 2:4] = np.exp(boxes_t[:, 2:4] * self.prior_scaling[1]) * default_boxes_t[:, 2:4]

        bboxes = np.zeros((len(boxes_t), 4), dtype=np.float32)

        bboxes[:, [0, 1]] = boxes_t[:, [0, 1]] - boxes_t[:, [2, 3]] / 2
        bboxes[:, [2, 3]] = boxes_t[:, [0, 1]] + boxes_t[:, [2, 3]] / 2

        return np.clip(bboxes, 0, 1)


grad_scale = ops.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.Reciprocal()(scale)


class TrainingWrapper(nn.Cell):
    """
    Encapsulation class of SSD network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        use_global_norm(bool): Whether apply global norm before optimizer. Default: False
    """

    def __init__(self, network, optimizer, sens=1.0, use_global_norm=False):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.use_global_norm = use_global_norm
        self.parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = ms.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = ms.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        self.hyper_map = ops.HyperMap()

    def construct(self, *args):
        """Construct method."""
        weights = self.weights
        loss = self.network(*args)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        if self.use_global_norm:
            grads = self.hyper_map(ops.partial(grad_scale, ops.scalar_to_array(self.sens)), grads)
            grads = ops.clip_by_global_norm(grads)
        self.optimizer(grads)
        return loss


def intersect(box_a, box_b):
    """Compute the intersect of two sets of boxes."""
    max_yx = np.minimum(box_a[:, 2:4], box_b[2:4])
    min_yx = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_yx - min_yx), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes."""
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union


def apply_eval(network, dataset, detection_engine):
    """Evaluation."""
    network.set_train(False)

    for data in dataset.create_dict_iterator(num_epochs=1):
        image_id = data["image_id"]
        image = data["image"]
        image_shape = data["image_shape"]

        output = network(image, image_id, image_shape)

        for batch_idx in range(image.shape[0]):
            pred_batch = {
                "boxes": output[0].asnumpy()[batch_idx],
                "box_scores": output[1].asnumpy()[batch_idx],
                "img_id": int(np.squeeze(image_id.asnumpy()[batch_idx])),
                "image_shape": image_shape.asnumpy()[batch_idx]
            }
            detection_engine.detect(pred_batch)

    detection_engine.get_eval_result()


def apply_nms(all_boxes, all_scores, thres, max_boxes):
    """Apply NMS to bboxes."""
    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]

    return keep
