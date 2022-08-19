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
# ==============================================================================
"""Load COCO dataset."""

import os
from typing import Optional, Any, Callable, List
import numpy as np

from PIL import Image
from pycocotools.coco import COCO

from mindvision.check_param import Validator
from mindvision.dataset.meta import Dataset, ParseDataset
from mindvision.io.path import check_file_exist, check_dir_exist
from mindvision.msdetection.utils.coco_utils import xywh2xyxy
from mindvision.msdetection.dataset.transforms import DetectionPad, DetectionHWC2CHW
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["COCODetection", "ParseCOCODetection"]


@ClassFactory.register(ModuleType.DATASET)
class COCODetection(Dataset):
    """COCO dataset."""

    def __init__(self,
                 path: str,
                 split: str = "train",
                 transforms: Optional[Callable] = None,
                 batch_size: int = 32,
                 repeat_num: int = 1,
                 shuffle: Optional[bool] = None,
                 num_parallel_workers: int = 1,
                 num_shards: Optional[int] = None,
                 shard_id: Optional[int] = None,
                 download: bool = False,
                 remove_invalid_annotations: bool = True,
                 filter_crowd_annotations: bool = True,
                 trans_record: bool = False) -> None:
        Validator.check_string(split, ["train", "val", "infer"], "split")

        if split == "val":
            schema_json = {
                "image": {"type": "bytes"},
                "image_id": {"type": "int32", "shape": [1]},
                "image_shape": {"type": "float32", "shape": [2]}
            }
            columns_list = ["image", "image_id", "image_shape"]
        else:
            schema_json = {
                "image": {"type": "bytes"},
                "boxes": {"type": "int32", "shape": [-1, 4]},
                "labels": {"type": "int32", "shape": [-1]}
            }
            columns_list = ["image", "boxes", "labels"]

        if download:
            raise ValueError("COCO dataset download is not supported.")

        if split != "infer":
            self.parse_coco = ParseCOCODetection(path=path,
                                                 split=split,
                                                 remove_invalid_annotations=remove_invalid_annotations,
                                                 filter_crowd_annotations=filter_crowd_annotations)
            load_data = self.parse_coco.parse_dataset

        super(COCODetection, self).__init__(path=path,
                                            split=split,
                                            load_data=load_data,
                                            transforms=transforms,
                                            batch_size=batch_size,
                                            repeat_num=repeat_num,
                                            shuffle=shuffle,
                                            num_parallel_workers=num_parallel_workers,
                                            num_shards=num_shards,
                                            shard_id=shard_id,
                                            columns_list=columns_list,
                                            schema_json=schema_json,
                                            trans_record=trans_record)

    @property
    def index2label(self):
        """Get the mapping of indexes and labels."""
        coco_classes = ParseCOCODetection.COCO_CLASSES
        index2label = {}
        for i, cls in enumerate(coco_classes):
            index2label[i] = cls
        return index2label

    def default_transform(self):
        """Default data augmentation."""
        return [DetectionPad(), DetectionHWC2CHW()]

    def pipelines(self):
        """Data augmentation."""
        trans = self.transforms if self.transforms else self.default_transform()
        self.dataset = self.dataset.map(operations=trans,
                                        input_columns=self.columns_list,
                                        num_parallel_workers=self.num_parallel_workers)


class ParseCOCODetection(ParseDataset):
    """Parse COCO2017 dataset."""
    COCO_CLASSES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                    'refrigerator', 'book', 'clock', 'vase', 'scissors',
                    'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, path: str,
                 split: str,
                 remove_invalid_annotations=True,
                 filter_crowd_annotations=True):
        super(ParseCOCODetection, self).__init__(path)
        self.img_root = os.path.join(path, "{}2017".format(split))
        check_dir_exist(self.img_root)
        anno_file = "instances_{}2017.json".format(split)
        self.anno_path = os.path.join(path, "annotations", anno_file)
        check_file_exist(self.anno_path)
        self.split = split
        self.coco = COCO(self.anno_path)

        self.class_dict = {}
        for i, cls in enumerate(self.COCO_CLASSES):
            self.class_dict[cls] = i

        self.categories = {
            cat["id"]: cat["name"] for cat in self.coco.cats.values()
        }

        self.remove_invalid_annotations = remove_invalid_annotations
        self.filter_crowd_annotations = filter_crowd_annotations

        self.img_ids, self.image_path_dict, self.image_anno_dict, self.image_labels_dict = self.__load_data()

    def __coco_remove_invalid_annotations(self, img_ids):
        """Remove images without annotations."""

        def has_only_empty_bbox(annotation):
            return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in annotation)

        def has_valid_annotation(annotation):
            """Check annotation file."""
            # if it's empty, there is no annotation
            if not annotation:
                return False
            # if all boxes have close to zero area, there is no annotation
            if has_only_empty_bbox(annotation):
                return False

            return True

        valid_ids = []
        for img_id in img_ids:
            annotation = self.__load_target(img_id)

            if has_valid_annotation(annotation):
                valid_ids.append(img_id)

        return valid_ids

    def __load_image(self, img_id: int) -> Image.Image:
        """Load image."""
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        return Image.open(os.path.join(self.img_root, path)).convert("RGB")

    def __load_target(self, img_id: int) -> List[Any]:
        """Load target."""
        annotation_ids = self.coco.getAnnIds(img_id, iscrowd=None)
        return self.coco.loadAnns(annotation_ids)

    def __load_data(self):
        """Load COCO dataset."""
        img_ids = self.coco.getImgIds()

        if self.remove_invalid_annotations:
            img_ids = self.__coco_remove_invalid_annotations(img_ids)

        image_path_dict = {}
        image_anno_dict = {}
        image_labels_dict = {}

        for img_id in img_ids:
            file_name = self.coco.loadImgs(img_id)[0]["file_name"]
            image_path = os.path.join(self.img_root, file_name)
            targets = self.__load_target(img_id)

            if self.filter_crowd_annotations:
                targets = [target for target in targets if target["iscrowd"] == 0]

            boxes = [target["bbox"] for target in targets]
            babels = [self.class_dict[self.categories[target["category_id"]]] for target in targets]

            out_boxes = []
            for bbox in boxes:
                bbox = xywh2xyxy(bbox)
                out_boxes.append(bbox)

            image_path_dict[img_id] = image_path
            image_anno_dict[img_id] = out_boxes
            image_labels_dict[img_id] = babels

        return img_ids, image_path_dict, image_anno_dict, image_labels_dict

    def parse_dataset(self, *args):
        """Parse COCO dataset."""
        if not args:
            return self.img_ids, self.image_path_dict, self.image_anno_dict, self.image_labels_dict

        img_id = self.img_ids[args[0]]
        image = self.__load_image(img_id)
        image_h, image_w, _ = image.shape
        boxes = self.image_anno_dict[img_id]
        labels = self.image_labels_dict[img_id]

        if self.split == "val":
            return image, img_id, np.array((image_h, image_w), np.float32)

        return image, boxes, labels
