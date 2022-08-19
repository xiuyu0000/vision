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
"""Test LeNet5 network for training in GPU."""

import os
import pytest

from tests.st.utils import parse_log_file


def lenet_train():
    """Test LeNet5 network."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = "/home/workspace/mindspore_dataset/mnist/"
    model_path = os.path.join(cur_path, "../../../../../examples/classification/lenet/")
    train_log = os.path.join(cur_path, "lenet_train_gpu.log")
    infer_log = os.path.join(cur_path, "lenet_infer_gpu.log")
    ckpt_file = "./lenet/lenet-10_1875.ckpt"

    # run train.
    exec_network_shell = "cd {0}; python lenet_mnist_train.py --data_url={1} > {2} 2>&1" \
        .format(model_path, dataset_path, train_log)
    ret = os.system(exec_network_shell)
    assert ret == 0

    # run validation.
    exec_network_shell = "cd {0}; python lenet_mnist_eval.py --data_url={1} --ckpt_file={2} > {3} 2>&1" \
        .format(model_path, dataset_path, ckpt_file, infer_log)
    ret = os.system(exec_network_shell)
    assert ret == 0

    pattern = r"'Top_1_Accuracy': (\d\.\d+)"
    acc = parse_log_file(pattern, infer_log)
    # print("acc is", acc)
    assert acc[0] > 0.98


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lenet_train_gpu():
    """
    Feature: test lenet network.
    Description: test lenet network for training.
    Expectation: success.
    """
    lenet_train()
