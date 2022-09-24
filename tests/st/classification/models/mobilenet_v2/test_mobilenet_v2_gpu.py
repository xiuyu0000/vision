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
"""Test MobileNetV2 network for training in GPU."""

import os
import pytest

from mindspore import nn
from mindspore import context
from mindspore.train import Model

from mindvision.classification.dataset import ImageNet
from mindvision.classification.models import mobilenet_v2
from mindvision.engine.loss import CrossEntropySmooth
from mindvision.check_param import Validator

cur_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = "/home/workspace/mindspore_dataset/imagenet/imagenet_original"
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mobilenet_v2_train_gpu():
    """
    Feature: test mobilenet_v2 network.
    Description: test mobilenet_v2 network for training.
    Expectation: success.
    """
    dataset = ImageNet(dataset_path,
                       split="train",
                       num_parallel_workers=8,
                       shuffle=True,
                       resize=96,
                       batch_size=32,
                       repeat_num=1)
    dataset_train = dataset.run()

    network = mobilenet_v2(num_classes=1001, alpha=0.5, pretrained=False, resize=96)
    lr = 0.1
    momentum = 0.9
    optimizer = nn.Momentum(network.trainable_params(), lr, momentum)
    criterion = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, classes_num=1001)
    net_with_criterion = nn.WithLossCell(network, criterion)
    train_network = nn.TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    losses = []

    dataset = dataset_train.create_dict_iterator()
    for _ in range(5):
        data = next(dataset)
        image = data["image"]
        label = data["label"]
        loss = train_network(image, label)
        losses.append(loss)
    Validator.check_equal_int(len(losses), 5)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mobilenet_v2_eval_gpu():
    """
    Feature: test mobilenet_v2 network.
    Description: test mobilenet_v2 network for eval.
    Expectation: success.
    """
    dataset = ImageNet(dataset_path,
                       split="val",
                       num_parallel_workers=8,
                       shuffle=True,
                       resize=96,
                       batch_size=32,
                       repeat_num=1)
    dataset_eval = dataset.run()

    network = mobilenet_v2(num_classes=1001, alpha=0.5, pretrained=True, resize=96)

    criterion = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, classes_num=1001)
    eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                    'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}
    model = Model(network, criterion, metrics=eval_metrics)
    result = model.eval(dataset_eval)

    assert result['Top_1_Accuracy'] > 0.52
