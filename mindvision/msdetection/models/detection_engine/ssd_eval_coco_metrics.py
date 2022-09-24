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
""" SSD COCO detection engine. """

import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mindvision.msdetection.dataset.coco import ParseCOCODetection
from mindvision.msdetection.models.utils.ssd_utils import apply_nms
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["SSDDetectionEngine"]


@ClassFactory.register(ModuleType.DETECTION_ENGINE)
class SSDDetectionEngine:
    """Detection Engine for SSD."""

    def __init__(self, num_classes, ann_file, min_score, nms_threshold, max_boxes):
        self.num_classes = num_classes
        self.ann_file = ann_file
        self.min_score = min_score
        self.nms_threshold = nms_threshold
        self.max_boxes = max_boxes
        self.classes = ParseCOCODetection.COCO_CLASSES

        self.val_cls_dict = {i: cls for i, cls in enumerate(self.classes)}
        self.coco_gt = COCO(self.ann_file)
        cat_ids = self.coco_gt.loadCats(self.coco_gt.getCatIds())
        self.class_dict = {cat["name"]: cat["id"] for cat in cat_ids}

        self.predictions = []
        self.img_ids = []

    def detect(self, batch):
        """Post process the detection results."""
        pred_boxes = batch['boxes']
        box_scores = batch['box_scores']
        img_id = batch['img_id']
        h, w = batch['image_shape']

        final_boxes = []
        final_label = []
        final_score = []
        self.img_ids.append(img_id)

        for c in range(1, self.num_classes):
            class_box_scores = box_scores[:, c]
            score_mask = class_box_scores > self.min_score
            class_box_scores = class_box_scores[score_mask]
            class_boxes = pred_boxes[score_mask] * [w, h, w, h]

            if score_mask.any():
                nms_index = apply_nms(class_boxes, class_box_scores, self.nms_threshold, self.max_boxes)
                class_boxes = class_boxes[nms_index]
                class_box_scores = class_box_scores[nms_index]

                final_boxes += class_boxes.tolist()
                final_score += class_box_scores.tolist()
                final_label += [self.class_dict[self.val_cls_dict[c]]] * len(class_box_scores)

        for loc, label, score in zip(final_boxes, final_label, final_score):
            res = {}
            res['image_id'] = img_id
            res['bbox'] = [loc[0], loc[1], loc[2] - loc[0], loc[3] - loc[1]]
            res['score'] = score
            res['category_id'] = label
            self.predictions.append(res)

    def get_eval_result(self):
        """Obtain the evaluation results."""
        with open('predictions.json', 'w') as f:
            json.dump(self.predictions, f)

        coco_dt = self.coco_gt.loadRes('predictions.json')
        eval_results = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
        eval_results.params.imgIds = self.img_ids
        eval_results.evaluate()
        eval_results.accumulate()
        eval_results.summarize()
        print("\n========================================\n")
