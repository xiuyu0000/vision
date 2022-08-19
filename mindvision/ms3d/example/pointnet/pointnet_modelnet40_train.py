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
""" PointNet training script."""

import argparse

import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindvision.ms3d.engine.ops.NLLLoss import NLLLoss

from mindvision.ms3d.dataset.modelnet40 import ModelNet40
from mindvision.ms3d.models.pointnet_cls import pointnet_cls
from mindvision.engine.callback import LossMonitor

set_seed(2)


def pointnet_train(args_opt):
    """PointNet train."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data Pipeline.
    dataset = ModelNet40(path=args_opt.data_url,
                         split="train",
                         batch_size=args_opt.batch_size,
                         repeat_num=args_opt.repeat_num,
                         shuffle=True,
                         resize=args_opt.resize,
                         download=args_opt.download)

    dataset_train = dataset.run()
    step_size = dataset_train.get_dataset_size()

    # Create model.
    network = pointnet_cls(num_classes=args_opt.num_classes, pretrained=args_opt.pretrained)

    # Set learning rate scheduler.
    if args_opt.lr_decay_mode == "cosine_decay_lr":
        lr = nn.cosine_decay_lr(min_lr=args_opt.min_lr,
                                max_lr=args_opt.max_lr,
                                total_step=args_opt.epoch_size * step_size,
                                step_per_epoch=step_size,
                                decay_epoch=args_opt.decay_epoch)
    elif args_opt.lr_decay_mode == "piecewise_constant_lr":
        lr = nn.piecewise_constant_lr(args_opt.milestone, args_opt.learning_rates)

    # Define optimizer.
    network_opt = nn.Adam(network.trainable_params(), lr, args_opt.momentum)

    # Define loss function.
    network_loss = NLLLoss(reduction="mean")

    # Define metrics.
    metrics = {'acc'}

    # Init the model.
    model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(save_checkpoint_steps=step_size,
                                   keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix='pointnet_cls',
                                    directory=args_opt.ckpt_save_dir,
                                    config=ckpt_config)

    # Begin to train.
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor(lr)],
                dataset_sink_mode=args_opt.dataset_sink_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointNet train.')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--epoch_size', type=int, default=250, help='Train epoch size.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size.')
    parser.add_argument('--repeat_num', type=int, default=1, help='Number of repeat.')
    parser.add_argument('--num_classes', type=int, default=40, help='Number of classification.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='Max number of checkpoint files.')
    parser.add_argument('--ckpt_save_dir', type=str, default="./pointnet_cls", help='Location of training outputs.')
    parser.add_argument("--learning_rates", type=list, default=None, help="A list of learning rates.")
    parser.add_argument("--lr_decay_mode", type=str, default="cosine_decay_lr", help="Learning rate decay mode.")
    parser.add_argument("--min_lr", type=float, default=0.00001, help="The min learning rate.")
    parser.add_argument("--max_lr", type=float, default=0.001, help="The max learning rate.")
    parser.add_argument("--decay_epoch", type=int, default=250, help="Number of decay epochs.")
    parser.add_argument("--milestone", type=list, default=None, help="A list of milestone.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the moving average.")
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='The dataset sink mode.')
    parser.add_argument('--resize', type=int, default=32, help='Resize the height and weight of picture.')
    parser.add_argument('--download', type=bool, default=False, help='Download ModelNet40 train dataset.')

    args = parser.parse_known_args()[0]
    pointnet_train(args)
