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
""" Load ShapeNet dataset."""

import os
import json
from typing import Callable, Optional, Union, Tuple
import subprocess
import shlex

import numpy as np
from mindvision.check_param import Validator
from mindvision.dataset.download import read_dataset
from mindvision.dataset.meta import Dataset, ParseDataset
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.ms3d.dataset.transforms import PcToTensor

__all__ = ["ShapeNet", "ParseShapeNet"]


@ClassFactory.register(ModuleType.DATASET)
class ShapeNet(Dataset):
    """
    A source dataset that reads, parses and augments the ShapeNet dataset.

    Args:
        path (str): The root directory of the ModelNet40 dataset or inference pointcloud.
        split (str): The dataset split, supports "train", "val", or "infer". Default: "train".
        transform (callable, optional):A function transform that takes in a pointcloud. Default: None.
        target_transform (callable, optional):A function transform that takes in a label. Default: None.
        batch_size (int): The batch size of dataset. Default: 64.
        repeat_num (int): The repeat num of dataset. Default: 1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default: None.
        num_parallel_workers (int, optional): The number of subprocess used to fetch the dataset
            in parallel.Default: None.
        num_shards (int, optional): The number of shards that the dataset will be divided. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        resize (Union[int, tuple]): The output size of the resized image. If size is an integer, the smaller edge of the
            image will be resized to this value with the same image aspect ratio. If size is a sequence of length 2,
            it should be (height, width). Default: 224.
        download (bool): Whether to download the dataset. Default: False.

    Raises:
        ValueError: If `split` is not 'train', 'test' or 'infer'.

    Examples:
        >>> from mindvision.ms3d.dataset import ShapeNet
        >>> dataset = ShapeNet("./data/shapenetcore_partanno_segmentation_benchmark_v0/", "train")
        >>> dataset = dataset.run()

    About ShapeNet dataset:
    This dataset provides part segmentation to a subset of ShapeNetCore models, containing ~16K models from 16 shape
    categories. The number of parts for each category varies from 2 to 6 and there are a total number of 50 parts.

    You can unzip the dataset files into this directory structure and read them by MindSpore Vision's API.

    .. code-block::

        ./shapenetcore_partanno_segmentation_benchmark_v0_normal/
        ├── 02691156
        │   ├── points
        │   │   ├── 1a04e3eab45ca15dd86060f189eb133.pts
        │   │   ├── 1a32f10b20170883663e90eaf6b4ca52.pts
        │   │   ├── ....
        │   ├── points_label
        │   │   ├── 1a04e3eab45ca15dd86060f189eb133.seg
        │   │   ├── 1a32f10b20170883663e90eaf6b4ca52.seg
        │   │   ├── ....
        │   └── seg_img
        │       ├── 1a04e3eab45ca15dd86060f189eb133.png
        │       ├── 1a32f10b20170883663e90eaf6b4ca52.png
        │       └── ....
        ├── 02773838
        │   ├── points
        │   ├── points_label
        │   └── seg_img
        ├── ....
        ├── train_test_split
        └── synsetoffset2category.txt

    Citation:

    .. code-block::

        @article{Yi16,
        Author = {Li Yi and Vladimir G. Kim and Duygu Ceylan and I-Chao Shen and Mengyan Yan and Hao Su and Cewu Lu and
        Qixing Huang and Alla Sheffer and Leonidas Guibas},
        Journal = {SIGGRAPH Asia},
        Title = {A Scalable Active Framework for Region Annotation in 3D Shape Collections},
        Year = {2016}
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
                 download: bool = False):
        Validator.check_string(split, ["train", "val", "infer"], "split")
        if split != "infer":
            self.parse_shapenet = ParseShapeNet(path=path, split=split)
            load_data = self.parse_shapenet.parse_dataset
        else:
            load_data = read_dataset

        super(ShapeNet, self).__init__(path=path,
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
                                       download=download)

    @property
    def index2label(self):
        """Get the mapping of indexes and labels. TO DO"""
        raise ValueError("ShapeNet dataset index2label is not supported.")

    def download_dataset(self, split):
        """Download the ShapeNet data if it doesn't exist."""
        if split == "infer":
            raise ValueError("Download is not supported for infer.")
        self.parse_shapeNet.download_and_extract_archive()

    def default_transform(self):
        """Set the default transform for ShapeNet dataset. TO DO"""
        trans = []

        if self.split == "train":
            trans = [
                PcToTensor(),
            ]
        return trans


class ParseShapeNet(ParseDataset):
    """
    Parse ShapeNet dataset.

    Args:
        path (str): The root directory of the ShapeNet dataset or inference pointcloud.
        split (str): The dataset split, supports "train", "val", or "infer". Default: "train".
        num_points(int): The number of points. Default: 1024.
        class_choice (str): Choose some classes to train. Default: None.

    """
    url_path = 'https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip'

    def __init__(self,
                 path: str,
                 split: str,
                 num_points: int = 1024,
                 class_choice: str = None):
        super(ParseShapeNet, self).__init__(path=path)
        self.path = path
        self.split = split
        self.npoints = num_points
        self.class_choice = class_choice
        self.catfile = os.path.join(self.path, 'synsetoffset2category.txt')
        self.cat = {}
        self.seg_classes = {}

    def __load_data(self):
        """Parse data from ShapeNet dataset file."""
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not self.class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in self.class_choice}
        id2cat = {v: k for k, v in self.cat.items()}

        meta = {}
        splitfile = os.path.join(self.path, 'train_test_split', 'shuffled_{}_file_list.json'.format(self.split))
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                meta[id2cat[category]].append((os.path.join(self.path, category, 'points', uuid + '.pts'),
                                               os.path.join(self.path, category, 'points_label', uuid + '.seg')))
        datapath = []
        for item in self.cat:
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

        classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), './utils/num_seg_classes'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, num_seg_classes)
        return datapath, classes

    def download_and_extract_archive(self):
        """Download the ShapeNet dataset if it doesn't exists."""
        path = self.path
        zipfile = os.path.join(path, os.path.basename(self.url_path))
        subprocess.check_call(
            shlex.split("curl {} -o {}".format(self.url_path, zipfile))
        )
        subprocess.check_call(
            shlex.split("unzip {} -d {}".format(zipfile, path))
        )
        subprocess.check_call(shlex.split("rm {}".format(zipfile)))

    def parse_dataset(self):
        """Parse data from ShapeNet dataset file."""
        data_list = []
        label_list = []
        datapath, _ = self.__load_data()

        for index in range(len(datapath)):
            fn = datapath[index]
            point_set = np.loadtxt(fn[1]).astype(np.float32)
            seg = np.loadtxt(fn[2]).astype(np.int64)
            choice = np.random.choice(len(seg), self.npoints, replace=True)
            point_set = point_set[choice, :]
            point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
            point_set = point_set / dist

            seg = seg[choice]
            point_set = np.array(point_set)
            seg = np.array(seg)
            data_list.append(point_set)
            label_list.append(seg)

        data = np.array(data_list)
        labels = np.array(label_list)
        return data, labels
