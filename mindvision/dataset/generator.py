# Copyright 2021
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
""" The generator dataset. """

from typing import Callable
import numpy as np
import cv2

from mindspore.mindrecord import FileWriter


class DatasetGenerator:
    """ Dataset generator for getting image path and its corresponding label. """

    def __init__(self, load_data: Callable):
        self.load_data = load_data

    def __getitem__(self, item):
        """Get the image and label for each item."""

        return self.load_data(item)

    def __len__(self):
        """Get the the size of dataset."""
        return len(self.load_data()[0])


class DatasetToMR:
    """Transform dataset to MindRecord."""

    def __init__(self, load_data, destination, split, partition_number, schema_json, shard_id):
        self.load_data = load_data
        self.partition_number = partition_number
        if shard_id is None:
            self.file_name = "{}/{}.mindrecord".format(destination, split)
        else:
            self.file_name = "{}/{}{}.mindrecord".format(destination, split, shard_id)
        self.writer = FileWriter(file_name=self.file_name,
                                 shard_num=partition_number,
                                 overwrite=True)
        self.schema_json = schema_json

    def trans_to_mr(self):
        """Execute transformation from dataset to MindRecord."""
        # Set the header size.
        self.writer.set_header_size(1 << 24)
        # Set the page size.
        self.writer.set_page_size(1 << 26)

        # Create the schema.
        self.writer.add_schema(self.schema_json)

        if list(self.schema_json.keys()) == ["image", "label"]:
            images, labels = self.load_data()
            if isinstance(images, np.ndarray) and isinstance(labels, np.ndarray):
                for data, label in zip(images, labels):
                    data = data[..., [2, 1, 0]] if data.shape[-1] == 3 else data
                    _, img = cv2.imencode('.jpeg', data)
                    data_list = [{"image": img.tobytes(), "label": int(label)}]
                    self.writer.write_raw_data(data_list)
            elif isinstance(images, list) and isinstance(labels, list):
                for data, label in zip(images, labels):
                    with open(data, 'rb') as f:
                        image_data = f.read()
                    data_list = [{"image": image_data, "label": int(label)}]
                    self.writer.write_raw_data(data_list)

        if list(self.schema_json.keys()) == ["image", "boxes", "labels"]:
            img_ids, image_path_dict, image_anno_dict, image_labels_dict = self.load_data()
            for img_id in img_ids:
                image_path = image_path_dict[img_id]
                try:
                    with open(image_path, 'rb') as f:
                        img = f.read()
                    boxes = np.array(image_anno_dict[img_id], dtype=np.int32)
                    labels = np.array(image_labels_dict[img_id], dtype=np.int32)
                    row = {"image": img, "boxes": boxes, "labels": labels}
                    self.writer.write_raw_data([row])
                except FileNotFoundError as e:
                    raise e

        if list(self.schema_json.keys()) == ["image", "image_id", "image_shape"]:
            img_ids, image_path_dict, image_anno_dict, image_labels_dict = self.load_data()
            for img_id in img_ids:
                image_path = image_path_dict[img_id]
                try:
                    with open(image_path, 'rb') as f:
                        img = f.read()
                    image_h, image_w, _ = cv2.imread(image_path).shape
                    image_shape = np.array([image_h, image_w], np.float32)
                    img_id = np.array([img_id], dtype=np.int32)
                    row = {"image": img, "image_id": img_id, "image_shape": image_shape}
                    self.writer.write_raw_data([row])
                except FileNotFoundError as e:
                    raise e

        self.writer.commit()

        return self.file_name
