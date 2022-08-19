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
"""X3D training script."""

import argparse

import mindspore.nn as nn
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from mindvision.check_param import Validator, Rel
from mindvision.msvideo.dataset import Kinetic400
from mindvision.engine.lr_schedule.lr_schedule import warmup_step_lr
from mindvision.msvideo.dataset.transforms import VideoRandomCrop, VideoRandomHorizontalFlip
from mindvision.msvideo.dataset.transforms import VideoResize, VideoToTensor
from mindvision.msvideo.models.x3d import x3d_m, x3d_l, x3d_s, x3d_xs


def x3d_kinetics400_train(args_opt):
    """X3D train"""
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target)

    # Data Pipeline.
    if args_opt.run_distribute:
        if args_opt.device_target == "Ascend":
            init()
        else:
            init("nccl")

        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        dataset = Kinetic400(args_opt.data_url,
                             split="train",
                             seq=args_opt.seq,
                             num_parallel_workers=args_opt.num_parallel_workers,
                             shuffle=True,
                             num_shards=device_num,
                             shard_id=rank_id,
                             batch_size=args_opt.batch_size,
                             repeat_num=args_opt.repeat_num)
        ckpt_save_dir = args_opt.ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"
    else:
        dataset = Kinetic400(args_opt.data_url,
                             split="train",
                             seq=args_opt.seq,
                             num_parallel_workers=args_opt.num_parallel_workers,
                             shuffle=True,
                             batch_size=args_opt.batch_size,
                             repeat_num=args_opt.repeat_num)
        ckpt_save_dir = args_opt.ckpt_save_dir

    # perpare dataset.
    transforms = [VideoResize([256, 256]),
                  VideoRandomCrop([224, 224]),
                  VideoRandomHorizontalFlip(0.5),
                  VideoToTensor()]
    dataset.transform = transforms
    dataset_train = dataset.run()
    Validator.check_int(dataset_train.get_dataset_size(), 0, Rel.GT)
    step_size = dataset_train.get_dataset_size()

    # Create model.
    if args_opt.model_name == "x3d_m":
        network = x3d_m(num_classes=args_opt.num_classes,
                        dropout_rate=args_opt.dropout_rate,
                        depth_factor=args_opt.depth_factor,
                        num_frames=args_opt.num_frames,
                        train_crop_size=args_opt.train_crop_size)
    elif args_opt.model_name == "x3d_s":
        network = x3d_s(num_classes=args_opt.num_classes,
                        dropout_rate=args_opt.dropout_rate,
                        depth_factor=args_opt.depth_factor,
                        num_frames=args_opt.num_frames,
                        train_crop_size=args_opt.train_crop_size)
    elif args_opt.model_name == "x3d_xs":
        network = x3d_xs(num_classes=args_opt.num_classes,
                         dropout_rate=args_opt.dropout_rate,
                         depth_factor=args_opt.depth_factor,
                         num_frames=args_opt.num_frames,
                         train_crop_size=args_opt.train_crop_size)
    elif args_opt.model_name == "x3d_l":
        network = x3d_l(num_classes=args_opt.num_classes,
                        dropout_rate=args_opt.dropout_rate,
                        depth_factor=args_opt.depth_factor,
                        num_frames=args_opt.num_frames,
                        train_crop_size=args_opt.train_crop_size)

    # Set lr scheduler.
    if args_opt.lr_decay_mode == 'exponential':
        learning_rate = warmup_step_lr(lr=args_opt.learning_rate,
                                       lr_epochs=args_opt.milestone,
                                       steps_per_epoch=step_size,
                                       warmup_epochs=args_opt.warmup_epochs,
                                       max_epoch=args_opt.epoch_size,
                                       gamma=args_opt.gamma)

    # Define optimizer.
    network_opt = nn.SGD(network.trainable_params(),
                         learning_rate,
                         weight_decay=args_opt.weight_decay)

    # Define loss function.
    network_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Define metrics.
    metrics = {'acc'}

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=step_size,
        keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix='x3d_kinetics400',
                                    directory=ckpt_save_dir,
                                    config=ckpt_config)

    # Init the model.
    model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)

    # Begin to train.
    print('[Start training `{}`]'.format('x3d_kinetics400'))
    print("=" * 80)
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor()],
                dataset_sink_mode=args_opt.dataset_sink_mode)
    print('[End of training `{}`]'.format('x3d_kinetics400'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='X3D train.')
    parser.add_argument('--device_target', type=str, default="GPU",
                        choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--run_distribute', type=bool, default=False,
                        help='Distributed parallel training.')
    parser.add_argument('--data_url', type=str, default="",
                        help='Location of data.')
    parser.add_argument('--seq', type=int, default=16,
                        help='Number of frames of captured video.')
    parser.add_argument('--num_parallel_workers', type=int, default=8,
                        help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of batch size.')
    parser.add_argument('--repeat_num', type=int, default=1,
                        help='Number of repeat.')
    parser.add_argument('--ckpt_save_dir', type=str, default="./x3d",
                        help='Location of training outputs.')
    parser.add_argument("--model_name", type=str, default="x3d_m",
                        help="Name of model.", choices=["x3d_m", "x3d_l", "x3d_s", "x3d_xs"])
    parser.add_argument('--num_classes', type=int, default=400,
                        help='Number of classification.')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--depth_factor', type=float, default=2.2,
                        help='Depth expansion factor.')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='The number of frames of the input clip.')
    parser.add_argument('--train_crop_size', type=int, default=224,
                        help='The spatial crop size for training.')
    parser.add_argument('--lr_decay_mode', type=str, default="exponential",
                        help='Learning rate decay mode.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--milestone', type=list, default=[15, 30, 75],
                        help='A list of milestone.')
    parser.add_argument('--warmup_epochs', type=int, default=35,
                        help='Warmup epochs.')
    parser.add_argument('--epoch_size', type=int, default=150,
                        help='Train epoch size.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Decay rate of learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.00005,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10,
                        help='Max number of checkpoint files.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False,
                        help='The dataset sink mode.')

    args = parser.parse_known_args()[0]
    x3d_kinetics400_train(args)
