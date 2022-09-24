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
"""Load AFLW2000-3D dataset."""

import os
from typing import Callable, Optional, Union, Tuple
from pathlib import Path
import subprocess
import shlex
import gdown
import numpy as np

import mindspore.dataset.vision.c_transforms as transforms

from mindvision.io.images import imread
from mindvision.dataset.download import read_dataset
from mindvision.check_param import Validator
from mindvision.ms3d.utils.synergynet_util import Crop
from mindvision.dataset.meta import Dataset, ParseDataset
from mindvision.engine.class_factory import ClassFactory, ModuleType

__all__ = ["AFLW2000", "ParseAFLW2000"]


@ClassFactory.register(ModuleType.DATASET)
class AFLW2000(Dataset):
    """
    A source dataset that reads, parses and augments the ModelNet40 dataset.

    The generated dataset has two columns :py:obj:`[image]`.
    The tensor of column :py:obj:`image` is a matrix of the float32 type.

    Args:
        path (str): The root directory of the AFLW2000 dataset or inference image.
        split (str): The dataset split, supports "train", "test", or "infer". Default: "test".
        transform (callable, optional):A function transform that takes in a image. Default: None.
        target_transform (callable, optional):A function transform that takes in a label. Default: None.
        batch_size (int): The batch size of dataset. Default: 64.
        resize (Union[int, tuple]): The output size of the resized image. If size is an integer, the smaller edge of the
            image will be resized to this value with the same image aspect ratio. If size is a sequence of length 2,
            it should be (height, width). Default: 224.
        repeat_num (int): The repeat num of dataset. Default: 1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Default: None.
        num_parallel_workers (int): The number of subprocess used to fetch the dataset in parallel. Default: 1.
        num_shards (int, optional): The number of shards that the dataset will be divided. Default: None.
        download (bool): Whether to download the dataset. Default: False.
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        download (bool) : Whether to download the dataset. Default: False.

    Raises:
        ValueError: If `split` is not 'train', 'test' or 'infer'.

    Examples:
        >>> from mindvision.ms3d.dataset import AFLW2000
        >>> dataset = AFLW2000('./data/aflw2000_data/','test')
        >>> dataset = dataset.run()

    About AFLW2000 dataset:
    The AFLW2000 dataset contains 2,000 face images .

    You can unzip the original AFLW2000 dataset files into this directory structure and read them by
    MindSpore Vision's API.

    .. code-block::

        ./aflw2000_data/
        ├── AFLW2000-3D_crop
        │   ├── image00002.jpg
        │   ├── image00004.jpg
        │   └── ....
        ├── eval
        │   ├── AFLW2000-3D.pose.npy
        │   ├── AFLW2000-3D.pts68.npy
        │   └── ....
        └── AFLW2000-3D_crop.list

    Citation:

    .. code-block::
        @inproceedings{7780392,
                    author={Zhu, Xiangyu and Lei, Zhen and Liu, Xiaoming and Shi, Hailin and Li, Stan Z.},
                    booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
                    title={Face Alignment Across Large Poses: A 3D Solution},
                    year={2016},
                    pages={146-155}
                }
    """

    def __init__(self,
                 path: str,
                 split: str = "test",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 batch_size: int = 128,
                 repeat_num: int = 1,
                 shuffle: Optional[bool] = None,
                 num_parallel_workers: int = 1,
                 num_shards: Optional[int] = None,
                 shard_id: Optional[int] = None,
                 resize: Union[int, Tuple[int, int]] = 224,
                 download: bool = False):
        Validator.check_string(split, ["train", "test", "infer"], "split")
        if split == "test":
            self.parse_aflw2000 = ParseAFLW2000(path=path)
            load_data = self.parse_aflw2000.parse_dataset
        else:
            load_data = read_dataset

        super(AFLW2000, self).__init__(path=path,
                                       split=split,
                                       load_data=load_data,
                                       transform=transform,
                                       target_transform=target_transform,
                                       batch_size=batch_size,
                                       repeat_num=repeat_num,
                                       shuffle=shuffle,
                                       num_parallel_workers=num_parallel_workers,
                                       num_shards=num_shards,
                                       shard_id=shard_id,
                                       resize=resize,
                                       download=download)

    @property
    def index2label(self):
        """Get the mapping of indexes and labels. TO DO"""
        raise ValueError("AFLW2000 dataset index2label is not supported.")

    def download_dataset(self, split):
        """Download the AFLW2000 data if it doesn't exist."""
        if split == "test":
            self.parse_aflw2000.download_and_extract_archive()
        raise ValueError("Download is not supported for train and infer.")

    def default_transform(self):
        """Set the default transform for AFLW2000 dataset. TO DO"""
        trans = []

        transpose = transforms.HWC2CHW()
        crop = Crop(5, mode='test')
        mean_channel = [127.5]
        std_channel = [128]
        normalize_op = transforms.Normalize(mean=mean_channel, std=std_channel)
        trans += [transpose, crop, normalize_op]
        return trans


class ParseAFLW2000(ParseDataset):
    """
    DownLoad and parse AFLW2000 dataset.

    Args:
        path (str): The root directory of the AFLW2000 dataset or inference image.
    """

    url_path = 'https://drive.google.com/uc?id=1SQsMhvAmpD1O8Hm0yEGom0C0rXtA0qs8'

    def __init__(self,
                 path: str):
        super(ParseAFLW2000, self).__init__(path=path)
        filelists = "./aflw2000_data/AFLW2000-3D_crop.list"
        self.lines = Path(filelists).read_text().strip().split('\n')

    def download_and_extract_archive(self):
        """Download the AFLW2000 dataset if it doesn't exists."""
        # Check whether the file exists
        filepath = os.path.split(self.path)[0]
        if os.path.isfile(filepath):
            return
        # download files
        gdown.download('https://drive.google.com/uc?id=1SQsMhvAmpD1O8Hm0yEGom0C0rXtA0qs8', quiet=False)
        zipfile = os.path.join(filepath, '.zip')
        subprocess.check_call(
            shlex.split("unzip {} -d {}".format(zipfile, filepath))
        )

        subprocess.check_call(shlex.split("rm {}".format(zipfile)))

    def parse_dataset(self):
        """Parse data from AFLW2000 dataset file."""
        path = os.path.join(self.path, "AFLW2000-3D_crop")
        data_list = []

        for index in range(len(self.lines)):
            datapath = os.path.join(path, self.lines[index])
            img = imread(datapath, mode="RGB")
            data_list.append(img)

        data = np.array(data_list)
        return data, data
