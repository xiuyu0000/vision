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
""" Create the CIFAR10 dataset. """

import os
import pickle
from typing import Optional, Callable, Union, Tuple
import numpy as np

import mindspore.dataset.vision.c_transforms as transforms

from mindvision.dataset.download import read_dataset
from mindvision.dataset.meta import Dataset, ParseDataset
from mindvision.check_param import Validator
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["Cifar10", "ParseCifar10"]


@ClassFactory.register(ModuleType.DATASET)
class Cifar10(Dataset):
    """
    A source dataset that downloads, reads, parses and augments the CIFAR-10 dataset.

    The generated dataset has two columns :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is a matrix of the float32 type.
    The tensor of column :py:obj:`label` is a scalar of the int32 type.

    Args:
        path (str): The root directory of the Cifar10 dataset or inference image.
        split (str): The dataset split supports "train", "test" or "infer". Default: "train".
        transform (callable, optional): A function transform that takes in a image. Default: None.
        target_transform (callable, optional): A function transform that takes in a label. Default: None.
        batch_size (int): The batch size of dataset. Default: 32.
        repeat_num (int): The repeat num of dataset. Default: 1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default: None.
        num_parallel_workers (int): The number of subprocess used to fetch the dataset in parallel. Default: 1.
        num_shards (int, optional): The number of shards that the dataset will be divided. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        resize (Union[int, tuple]): The output size of the resized image. If size is an integer, the smaller edge of the
            image will be resized to this value with the same image aspect ratio. If size is a sequence of length 2,
            it should be (height, width). Default: 224.
        download (bool) : Whether to download the dataset. Default: False.
        trans_to_mindrecord (bool): Whether transform dataset to MindRecord. Default: False.

    Raise:
        ValueError: If `split` is not 'train', 'test' or 'infer'.

    Examples:
        >>> from mindvision.classification.dataset import Cifar10
        >>> dataset = Cifar10("./data/", "train")
        >>> dataset = dataset.run()

    About CIFAR-10 dataset:

    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
    with 6000 images per class. There are 50000 training images and 10000 test images.
    The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

    Here is the original CIFAR-10 dataset structure.
    You can unzip the dataset files into the following directory structure and read them by MindSpore Vision's API.

    .. code-block::

        .
        └── cifar-10-batches-py
             ├── data_batch_1
             ├── data_batch_2
             ├── data_batch_3
             ├── data_batch_4
             ├── data_batch_5
             ├── test_batch
             ├── readme.html
             └── batches.meta

    Citation:

    .. code-block::

        @techreport{Krizhevsky09,
        author       = {Alex Krizhevsky},
        title        = {Learning multiple layers of features from tiny images},
        institution  = {},
        year         = {2009},
        howpublished = {http://www.cs.toronto.edu/~kriz/cifar.html}
        }
    """

    def __init__(self,
                 path: str,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 batch_size: int = 32,
                 repeat_num: int = 1,
                 shuffle: Optional[bool] = None,
                 num_parallel_workers: int = 1,
                 num_shards: Optional[int] = None,
                 shard_id: Optional[int] = None,
                 resize: Union[int, Tuple[int, int]] = 224,
                 download: bool = False,
                 trans_record: bool = False):
        Validator.check_string(split, ["train", "test", "infer"], "split")
        mode = "RGB"
        schema_json = {"image": {"type": "bytes"}, "label": {"type": "int64"}}
        columns_list = ["image", "label"]

        if split == "infer" and download:
            raise ValueError("Download is not supported for infer.")

        if split != "infer":
            self.parse_cifar10 = ParseCifar10(path=os.path.join(path, split),
                                              shard_id=shard_id, download=download)
            load_data = self.parse_cifar10.parse_dataset
        else:
            load_data = read_dataset

        super(Cifar10, self).__init__(path=path,
                                      split=split,
                                      load_data=load_data,
                                      transform=transform,
                                      target_transform=target_transform,
                                      batch_size=batch_size,
                                      repeat_num=repeat_num,
                                      resize=resize,
                                      shuffle=shuffle,
                                      num_parallel_workers=num_parallel_workers,
                                      num_shards=num_shards,
                                      shard_id=shard_id,
                                      mode=mode,
                                      columns_list=columns_list,
                                      schema_json=schema_json,
                                      trans_record=trans_record
                                      )

    @property
    def index2label(self):
        """Get the mapping of indexes and labels."""
        return ParseCifar10.load_meta(self.path)

    def default_transform(self):
        """Set the default transform for Cifar10 dataset."""
        trans = []

        if self.trans_record:
            trans += [transforms.Decode()]

        if self.split == "train":
            trans += [
                transforms.RandomCrop((32, 32), (4, 4, 4, 4)),
                transforms.RandomHorizontalFlip(prob=0.5)
            ]

        trans += [
            transforms.Resize(self.resize),
            transforms.Rescale(1.0 / 255.0, 0.0),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            transforms.HWC2CHW()
        ]

        return trans


class ParseCifar10(ParseDataset):
    """
    DownLoad and parse Cifar10 dataset.

    Args:
        path (str): The root path of the Cifar10 dataset join train or test.

    Examples:
        >>> parse_data = ParseCifar10("./cifar10/train")
    """

    url_path = {"path": "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                "md5": "c58f30108f718f92721af3b95e74349a"}
    base_dir = "cifar-10-batches-py"
    classes_key = "label_names"

    extract = {
        "train": [
            ("data_batch_1", "c99cafc152244af753f735de768cd75f"),
            ("data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"),
            ("data_batch_3", "54ebc095f3ab1f0389bbae665268c751"),
            ("data_batch_4", "634d18415352ddfa80567beed471001a"),
            ("data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"),
        ],
        "test": [
            ("test_batch", "40351d587109b95175f43aff81a1287e")
        ],
        "meta": [
            ("batches.meta", "5ff9c542aee3614f3951f8cda6e48888")
        ]
    }

    def __init__(self, path: str, shard_id: Optional[int] = None, download: bool = False):
        super(ParseCifar10, self).__init__(path, shard_id)
        self.path, self.basename = os.path.split(self.path)

        if download:
            self.download_and_extract_archive()

        self.data, self.labels = self.__load_data()

    def download_and_extract_archive(self):
        """Download the Cifar10 dataset if it doesn't exists."""
        bool_list = []
        # Check whether the file exists and check value of md5.
        for value in self.extract.values():
            for i in value:
                filename, md5 = i[0], i[1]
                file_path = os.path.join(self.path, self.base_dir, filename)
                bool_list.append(
                    os.path.isfile(file_path) and self.download.check_md5(file_path, md5)
                )

        if all(bool_list):
            return

        if self.shard_id is not None:
            self.path = os.path.join(self.path, 'dataset_{}'.format(str(self.shard_id)))

        # download files
        self.download.download_and_extract_archive(self.url_path["path"],
                                                   download_path=self.path,
                                                   md5=self.url_path["md5"])

    @classmethod
    def load_meta(cls, path):
        """Load meta file."""
        meta_file = cls.extract["meta"][0][0]
        meta_md5 = cls.extract["meta"][0][1]
        meta_path = os.path.join(path, cls.base_dir, meta_file)

        if not os.path.isfile(meta_path) and self.download.check_md5(meta_path, meta_md5):
            raise RuntimeError(
                "Metadata file not found or check md5 value is incorrect. You can set download=True.")

        with open(meta_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            classes = data[cls.classes_key]
            index2label = {i: v for i, v in enumerate(classes)}
            return index2label

    def __load_cifar_batch(self):
        """Load single batch of cifar."""
        if not os.path.isfile(self.data_path) and self.download.check_md5(self.data_path, self.md5):
            raise RuntimeError(
                "Dataset file not found or check md5 value is incorrect. You can set download=True.")
        with open(self.data_path, "rb") as f:
            data_dict = pickle.load(f, encoding="latin1")

        data = data_dict["data"]
        labels = data_dict["labels"] if "labels" in data_dict else data_dict["fine_labels"]

        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(labels, dtype=np.int32)

        return data, labels

    def __load_data(self):
        """Parse data from Cifar10 dataset file."""
        data_list = []
        labels_list = []
        file_list = self.extract[self.basename]

        for file_name, md5 in file_list:
            self.data_path = os.path.join(self.path, self.base_dir, file_name)
            self.md5 = md5
            data, labels = self.__load_cifar_batch()
            data_list.append(data)
            labels_list.append(labels)

        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        return data, labels

    def parse_dataset(self, *args):
        """Parse dataset."""
        if not args:
            return self.data, self.labels

        return self.data[args[0]], self.labels[args[0]]
