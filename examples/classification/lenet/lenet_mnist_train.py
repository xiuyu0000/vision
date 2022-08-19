# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#:
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" LeNet training script."""

import argparse

from mindspore import nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet
from mindvision.engine.callback import LossMonitor

set_seed(1)


def lenet_train(args_opt):
    """LeNet train."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data Pipeline.
    dataset = Mnist(args_opt.data_url,
                    split="train",
                    batch_size=args_opt.batch_size,
                    repeat_num=args_opt.repeat_num,
                    shuffle=True,
                    resize=args_opt.resize,
                    download=args_opt.download)

    dataset_train = dataset.run()
    step_size = dataset_train.get_dataset_size()

    # Create model.
    network = lenet(args_opt.num_classes, pretrained=args_opt.pretrained)

    # Define optimizer.
    network_opt = nn.Momentum(network.trainable_params(), args_opt.learning_rate, args_opt.momentum)

    # Define loss function.
    network_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Define metrics.
    metrics = {'acc'}

    # Init the model.
    model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=step_size,
        keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix='lenet',
                                    directory=args_opt.ckpt_save_dir,
                                    config=ckpt_config)

    # Begin to train.
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor(args_opt.learning_rate)],
                dataset_sink_mode=args_opt.dataset_sink_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LeNet train.')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--epoch_size', type=int, default=10, help='Train epoch size.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--batch_size', type=int, default=32, help='Number of batch size.')
    parser.add_argument('--repeat_num', type=int, default=1, help='Number of repeat.')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classification.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='Max number of checkpoint files.')
    parser.add_argument('--ckpt_save_dir', type=str, default="./lenet", help='Location of training outputs.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Value of learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the moving average.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=True, help='The dataset sink mode.')
    parser.add_argument('--resize', type=int, default=32, help='Resize the height and weight of picture.')
    parser.add_argument('--download', type=bool, default=False, help='Download Mnist train dataset.')

    args = parser.parse_known_args()[0]
    lenet_train(args)
