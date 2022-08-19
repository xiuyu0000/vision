# Copyright 2022 Huawei Technologies Co., Ltd
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
""" ConvNext training script. """

import os
import argparse

from mindspore import nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindvision.classification.dataset import ImageNet
from mindvision.classification.utils import get_group, ParamLRValueAssigner
from mindvision.engine.lr_schedule.lr_schedule import warmup_cosine_annealing_lr_v1

set_seed(1)


def set_device(args_opt):
    """Set device and ParallelMode(if device_num > 1)"""
    rank = 0
    device_target = args_opt.device_target
    device_num = int(os.environ.get("DEVICE_NUM", "1"))

    if device_num > 1:
        if device_target == "Ascend":
            context.set_context(device_id=int(os.environ["DEVICE_ID"]))
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        rank = get_rank()
    else:
        context.set_context(device_id=args_opt.device_id)


    return rank


def convnext_train(args_opt):
    """ ConvNext train."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data Pipeline.
    if args_opt.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
        rank = set_device(args_opt)
        ckpt_save_dir = args_opt.ckpt_save_dir + "_ckpt_" + str(rank) + "/"
    if args_opt.run_distribute and args_opt.device_target == "GPU":
        context.set_context(device_id=args_opt.device_id)
        init("nccl")
        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        dataset = ImageNet(args_opt.data_url,
                           split="train",
                           num_parallel_workers=args_opt.num_parallel_workers,
                           shuffle=True,
                           resize=args_opt.resize,
                           num_shards=device_num,
                           shard_id=rank_id,
                           batch_size=args_opt.batch_size,
                           repeat_num=args_opt.repeat_num)
        ckpt_save_dir = args_opt.ckpt_save_dir + "_ckpt_" + str(rank_id) + "/"
    else:
        dataset = ImageNet(args_opt.data_url,
                           split="train",
                           num_parallel_workers=args_opt.num_parallel_workers,
                           shuffle=True,
                           resize=args_opt.resize,
                           batch_size=args_opt.batch_size,
                           repeat_num=args_opt.repeat_num)
        ckpt_save_dir = args_opt.ckpt_save_dir

    dataset_train = dataset.run()
    step_size = dataset_train.get_dataset_size()

    # Create_model.
    if args_opt.model == 'convnext_tiny':
        from mindvision.classification.models import convnext_tiny
        network = convnext_tiny(pretrained=args_opt.pretrained, num_classes=args_opt.num_classes)
    if args_opt.model == 'convnext_small':
        from mindvision.classification.models import convnext_small
        network = convnext_small(pretrained=args_opt.pretrained, num_classes=args_opt.num_classes)
    if args_opt.model == 'convnext_base':
        from mindvision.classification.models import convnext_base
        network = convnext_base(pretrained=args_opt.pretrained, num_classes=args_opt.num_classes)
    if args_opt.model == 'convnext_large':
        from mindvision.classification.models import convnext_large
        network = convnext_large(pretrained=args_opt.pretrained, num_classes=args_opt.num_classes)
    if args_opt.model == 'convnext_xlarge':
        from mindvision.classification.models import convnext_xlarge
        network = convnext_xlarge(pretrained=args_opt.pretrained, num_classes=args_opt.num_classes)

    # Set lr scheduler.
    lr_scheduler = warmup_cosine_annealing_lr_v1(lr=args_opt.lr, steps_per_epoch=step_size,
                                                 warmup_epochs=20, max_epoch=args_opt.epoch_size,
                                                 t_max=150, eta_min=0)

    # group parameters
    if args_opt.lr_layer_scale < 1.0:
        num_layers = 12
        lr_scale_values = list(args_opt.lr_layer_scale ** (num_layers + 1 - i) for i in range(num_layers + 2))
        assigner = ParamLRValueAssigner(lr_scale_values)
    else:
        assigner = None
    params = get_group(network=network,
                       init_lr=args_opt.lr,
                       warmup_epochs=args_opt.warmup_epochs,
                       epoch_size=args_opt.epoch_size,
                       lr_scheduler=warmup_cosine_annealing_lr_v1,
                       step_per_epoch=step_size,
                       assigner=assigner,
                       weight_decay=args_opt.weight_decay)
    # Define optimizer
    network_opt = nn.AdamWeightDecay(params=params,
                                     learning_rate=lr_scheduler,
                                     beta1=0.9,
                                     beta2=0.999,
                                     eps=1e-6,
                                     weight_decay=0.05)
    # Define loss function
    network_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Define metrics
    metrics = {'acc'}

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=step_size,
        keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix=args_opt.model,
                                    directory=ckpt_save_dir,
                                    config=ckpt_config)

    # Init the model.
    model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)

    # Begin to train.
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor(), TimeMonitor()],
                dataset_sink_mode=args_opt.dataset_sink_mode)


def parse_args():
    """train args"""
    parser = argparse.ArgumentParser(description="ConvNext train.")
    parser.add_argument('--model', required=True, default=None,
                        choices=["convnext_tiny",
                                 "convnext_small",
                                 "convnext_base",
                                 "convnext_large",
                                 "convnext_xlarge"])
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--device_id', type=int, default=0, help='The machine number on which the code is run')
    parser.add_argument("--device_num", default=1, type=int, help="device num")
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--epoch_size', type=int, default=90, help='Train epoch size.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='Max number of checkpoint files.')
    parser.add_argument('--ckpt_save_dir', type=str, default="./resnet", help='Location of training outputs.')
    parser.add_argument('--num_parallel_workers', type=int, default=8, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size.')
    parser.add_argument('--repeat_num', type=int, default=1, help='Number of repeat.')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classification.')
    parser.add_argument('--lr_decay_mode', type=str, default="cosine_annealing_lr_v1", help='Learning rate decay mode.')
    parser.add_argument('--lr', type=str, default=4e-3, metavar='LR', help='learning rate.')
    parser.add_argument('--lr_layer_scale', type=float, default=1.0, help='learning rate decay for layers.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay value.')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='Number of classification.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=True, help='The dataset sink mode.')
    parser.add_argument('--run_distribute', type=bool, default=True, help='Run distribute.')
    parser.add_argument('--resize', type=int, default=224, help='Resize the image.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convnext_train(args)
