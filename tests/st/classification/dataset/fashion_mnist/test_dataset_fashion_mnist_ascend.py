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
"""Test FashionMnist dataset operators."""

import os
import pytest

import mindspore.dataset as ds

from mindvision.classification.dataset import FashionMnist

data_dir = "/home/workspace/mindspore_dataset/mindvision_data/fashion_mnist/"
cur_path = os.path.dirname(os.path.abspath(__file__))
download_dir = os.path.join(cur_path, "fashion_mnist")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fashion_mnist_split():
    """
    Feature: test class of FashionMnist.
    Description: test the split parameters of FashionMnist.
    Expectation: success.
    """
    data_train = FashionMnist(path=data_dir,
                              split="train")
    data_train = data_train.run()
    assert isinstance(data_train, ds.RepeatDataset)

    data_eval = FashionMnist(path=data_dir,
                             split="test")
    data_eval = data_eval.run()
    assert isinstance(data_eval, ds.RepeatDataset)

    data_infer = FashionMnist(path=cur_path,
                              split="infer",
                              batch_size=1)
    data_infer = data_infer.run()
    assert isinstance(data_infer, ds.RepeatDataset)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fasihon_mnist_batch_size():
    """
    Feature: test class of FashionMnist.
    Description: test the batch_size parameters of FashionMnist.
    Expectation: success.
    """
    data_train = FashionMnist(path=data_dir,
                              batch_size=64)
    data_train = data_train.run()
    step_size = data_train.get_dataset_size()
    assert data_train.get_batch_size() == 64
    assert step_size == 937


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mnist_repeat():
    """
    Feature: test class of FashionMnist.
    Description: test the repeat parameters of FashionMnist.
    Expectation: success.
    """
    data_train = FashionMnist(path=data_dir,
                              repeat_num=2)
    data_train = data_train.run()
    step_size = data_train.get_dataset_size()
    assert step_size == 3750


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fashion_mnist_resize():
    """
    Feature: test class of FashionMnist.
    Description: test the resize parameters of FashionMnist.
    Expectation: success.
    """
    data_train = FashionMnist(path=data_dir,
                              resize=64)
    data_train = data_train.run()
    data_iter = next(data_train.create_dict_iterator(output_numpy=True))
    images = data_iter["image"]
    assert images.shape == (32, 1, 64, 64)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fashion_mnist_download():
    """
    Feature: test class of FashionMnist.
    Description: test the download parameters of FashionMnist.
    Expectation: success.
    """
    data_train = FashionMnist(path=download_dir,
                              split="train",
                              download=True)
    data_train = data_train.run()
    assert isinstance(data_train, ds.RepeatDataset)

    data_eval = FashionMnist(path=download_dir,
                             split="test",
                             download=True)
    data_eval = data_eval.run()
    assert isinstance(data_eval, ds.RepeatDataset)

    error_msg = "Download is not supported for infer."
    with pytest.raises(ValueError, match=error_msg):
        FashionMnist(path=download_dir,
                     split="infer",
                     download=True)
