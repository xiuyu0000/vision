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
"""Load ModelNet40_Base dataset."""

import os
from typing import Callable, Optional, Union, Tuple
import subprocess
import shlex

import numpy as np
from mindvision.check_param import Validator
from mindvision.dataset.download import read_dataset
from mindvision.dataset.meta import Dataset, ParseDataset
from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.ms3d.dataset.transforms import PcToTensor, PcScale, PcShift

__all__ = ["ModelNet40", "ParseModelNet40"]


@ClassFactory.register(ModuleType.DATASET)
class ModelNet40(Dataset):
    """
    A source dataset that reads, parses and augments the ModelNet40 dataset.

    The generated dataset has three columns :py:obj:`[pointcloud, pointcloud_id, label]`.
    The tensor of column :py:obj:`pointcloud` is a 1024*3 matrix of the float32 type.
    The tensor of column :py:obj:`pointcloud_id` is int of the int32 type.
    The tensor of column :py:obj:`label` is a scalar of the int32 type.

    Args:
        path (str): The root directory of the ModelNet40 dataset or inference pointcloud.
        split (str): The dataset split, supports "train", "val", or "infer". Default: "train".
        transform (callable, optional):A function transform that takes in a pointcloud. Default: None.
        target_transform (callable, optional):A function transform that takes in a label. Default: None.
        batch_size (int): The batch size of dataset. Default: 64.
        resize (Union[int, tuple]): The output size of the resized image. If size is an integer, the smaller edge of the
            image will be resized to this value with the same image aspect ratio. If size is a sequence of length 2,
            it should be (height, width). Default: 224.
        repeat_num (int): The repeat num of dataset. Default: 1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default: None.
        download (bool): Whether to download the dataset. Default: False.
        mr_file (str, optional): The path of mindrecord files. Default: False.
        columns_list (tuple): The column names of output data. Default: ('image', 'image_id', "label").
        num_parallel_workers (int, optional): The number of subprocess used to fetch the dataset
            in parallel.Default: None.
        num_shards (int, optional): The number of shards that the dataset will be divided. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.

    Raises:
        ValueError: If `split` is not 'train', 'test' or 'infer'.

    Examples:
        >>> from mindvision.ms3d.dataset import ModelNet40
        >>> dataset = ModelNet40("./data/ModelNet40/", "train")
        >>> dataset = dataset.run()

    About ModelNet40 dataset:
    The ModelNet40 dataset contains 12,311 CAD models with 40 object categories.
    They are split into 9,843 models for training and 2,468 for testing.

    You can unzip the original ModelNet40 dataset files into this directory structure and read them by
    MindSpore Vision's API.

    .. code-block::

        ./ModelNet40/
        ├── airplane
        │   ├── airplane_0001.txt
        │   ├── airplane_0002.txt
        │   └── ....
        ├── bathtub
        │   ├── bathtub_0001.txt
        │   ├── bathtub_0002.txt
        │   └── ....
        ├── filelist.txt
        ├── modelnet40_shape_names.txt
        ├── modelnet40_test.txt
        └── modelnet40_train.txt
    Citation:

    .. code-block::
        @inproceedings{wu2015modelnet40,
                    title={3d shapenets: A deep representation for volumetric shapes},
                    author={Wu, Zhirong and Song, Shuran and Khosla, Aditya and Yu, Fisher and Zhang, Linguang and Tang,
                            Xiaoou and Xiao, Jianxiong},
                    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
                    pages={1912--1920},
                    year={2015}
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
                 resize: Union[int, Tuple[int, int]] = 1024,
                 download: bool = False,
                 use_norm: bool = False):
        Validator.check_string(split, ["train", "val", "infer"], "split")
        if split != "infer":
            self.parse_modelnet40 = ParseModelNet40(path=path, split=split, num_points=resize, use_norm=use_norm)
            load_data = self.parse_modelnet40.parse_dataset
        else:
            load_data = read_dataset

        super(ModelNet40, self).__init__(path=path,
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
        raise ValueError("ModelNet40 dataset index2label is not supported.")
        # return self.parse_modelnet40.index2label

    def download_dataset(self, split):
        """Download the ModelNet40 data if it doesn't exist."""
        if split == "infer":
            raise ValueError("Download is not supported for infer.")
        self.parse_modelnet40.download_and_extract_archive()

    def default_transform(self):
        """Set the default transform for ModelNet40 dataset. TO DO"""
        if self.split == "train":
            trans = [
                PcScale(scale_low=0.8, scale_high=1.2),
                PcShift(shift_range=0.1),
                PcToTensor(),
            ]
        else:
            trans = [
                PcToTensor(),
            ]
        return trans


class ParseModelNet40(ParseDataset):
    """
    Parse ModelNet40 dataset.

    Args:
        path (str): The root directory of the ModelNet40 dataset or inference pointcloud.
        split (str): The dataset split, supports "train", "val", or "infer". Default: "train".
    """
    url_path = 'https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip'

    def __init__(self,
                 path: str,
                 split: str,
                 num_points: int,
                 use_norm: bool):
        super(ParseModelNet40, self).__init__(path=path)
        self.split = split
        self.num_points = num_points
        self.use_norm = use_norm

    def extract_path(self):
        shapeid_path = "modelnet40_train.bak.txt" if self.split == "train" else "modelnet40_test.txt"
        catfile = os.path.join(self.path, "modelnet40_shape_names.txt")
        cat = [line.rstrip() for line in open(catfile)]
        classes = dict(zip(cat, range(len(cat))))
        return shapeid_path, classes

    def pc_normalize(self, data):
        centroid = np.mean(data, axis=0)
        data = data - centroid
        m = np.max(np.sqrt(np.sum(data ** 2, axis=1)))
        data /= m
        return data

    def download_and_extract_archive(self):
        """Download the ModelNet40 dataset if it doesn't exists."""
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
        """Parse data from ModelNet40 dataset file."""
        data_list = []
        label_list = []
        shapeid_path, classes = self.extract_path()
        shape_ids = [line.rstrip() for line in open(os.path.join(self.path, shapeid_path))]
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids]
        datapath = [(shape_names[i], os.path.join(self.path, shape_names[i], shape_ids[i]) + '.txt') for i
                    in range(len(shape_ids))]

        for index in range(len(datapath)):
            fn = datapath[index]
            label = classes[datapath[index][0]]
            label = np.array([label]).astype(np.int32)
            point_cloud = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.use_norm:
                point_cloud = point_cloud[:self.num_points, :]
            else:
                point_cloud = point_cloud[:self.num_points, :3]
            point_cloud[:, :3] = self.pc_normalize(point_cloud[:, :3])
            data_list.append(point_cloud)
            label_list.append(label)

        data = np.array(data_list)
        labels = np.concatenate(label_list, axis=0)
        return data, labels
