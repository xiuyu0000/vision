# Copyright 2021
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
""" DenseNet training script. """

import argparse

import mindspore.train
from mindspore.common import set_seed
from mindspore import context, nn
import mindspore.communication as comm
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindvision.engine.callback import LossMonitor, ValAccMonitor
from mindvision.classification.dataset import ImageNet

set_seed(1)


def densenet_train(args_opt):
    """Densenet train."""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target)

    # distributed
    if args_opt.run_distribute:
        comm.init("nccl")
        rank_id = comm.get_rank()
        device_num = comm.get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                          gradient_mean=True
                                          )
        dataset_train = ImageNet(args_opt.data_url,
                                 split="train",
                                 num_parallel_workers=args_opt.num_parallel_workers,
                                 shuffle=True,
                                 resize=args_opt.resize,
                                 num_shards=device_num,
                                 shard_id=rank_id,
                                 batch_size=args_opt.batch_size,
                                 repeat_num=args_opt.repeat_num)
        dataset_val = ImageNet(args_opt.data_url,
                               split="val",
                               num_parallel_workers=args_opt.num_parallel_workers,
                               shuffle=True,
                               resize=args_opt.resize,
                               num_shards=device_num,
                               shard_id=rank_id,
                               batch_size=args_opt.batch_size,
                               repeat_num=args_opt.repeat_num)
        ckpt_save_dir = args_opt.ckpt_save_dir + "_ckpt_" + str(rank_id) + "/"
    else:
        dataset_train = ImageNet(args_opt.data_url,
                                 split="train",
                                 num_parallel_workers=args_opt.num_parallel_workers,
                                 shuffle=True,
                                 resize=args_opt.resize,
                                 batch_size=args_opt.batch_size,
                                 repeat_num=args_opt.repeat_num)
        dataset_val = ImageNet(args_opt.data_url,
                               split="val",
                               num_parallel_workers=args_opt.num_parallel_workers,
                               shuffle=True,
                               resize=args_opt.resize,
                               batch_size=args_opt.batch_size,
                               repeat_num=args_opt.repeat_num)
        ckpt_save_dir = args_opt.ckpt_save_dir

    dataset_train = dataset_train.run()
    dataset_val = dataset_val.run()
    step_size = dataset_train.get_dataset_size()

    # Create model.
    if args_opt.model == 'densenet121':
        from mindvision.classification.models import densenet121
        network = densenet121(args_opt.num_classes,
                              pretrained=args_opt.pretrained)
    elif args_opt.model == 'densenet161':
        from mindvision.classification.models import densenet161
        network = densenet161(args_opt.num_classes,
                              pretrained=args_opt.pretrained)
    elif args_opt.model == 'densenet169':
        from mindvision.classification.models import densenet169
        network = densenet169(args_opt.num_classes,
                              pretrained=args_opt.pretrained)
    elif args_opt.model == 'densenet201':
        from mindvision.classification.models import densenet201
        network = densenet201(args_opt.num_classes,
                              pretrained=args_opt.pretrained)
    elif args_opt.model == 'densenet232':
        from mindvision.classification.models import densenet232
        network = densenet232(args_opt.num_classes,
                              pretrained=args_opt.pretrained)
    elif args_opt.model == 'densenet264':
        from mindvision.classification.models import densenet264
        network = densenet264(args_opt.num_classes,
                              pretrained=args_opt.pretrained)

    if args_opt.lr_decay_mode == 'piecewise_constant_lr':
        lr = nn.piecewise_constant_lr(
            [m * step_size for m in args_opt.milestone], args_opt.learning_rate)
    elif args_opt.lr_decay_mode == 'cosine_decay_lr':
        lr = nn.cosine_decay_lr(min_lr=args_opt.min_lr, max_lr=args_opt.max_lr,
                                total_step=args_opt.epoch_size * step_size, step_per_epoch=step_size,
                                decay_epoch=args_opt.decay_epoch)

    network_opt = nn.Momentum(
        network.trainable_params(), lr, args_opt.momentum)
    network_loss = nn.SoftmaxCrossEntropyWithLogits(
        sparse=True, reduction='mean')
    metrics = {'acc'}

    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=step_size,
        keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix=args_opt.model,
                                    directory=ckpt_save_dir,
                                    config=ckpt_config)

    model = mindspore.train.Model(
        network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)
    print("start training")
    model.train(args_opt.epoch_size, dataset_train,
                callbacks=[ckpt_callback, LossMonitor(lr, per_print_times=1000),
                           ValAccMonitor(model, dataset_val, num_epochs=1, metric_name='acc')],
                dataset_sink_mode=args_opt.dataset_sink_mode
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DenseNet train.')
    parser.add_argument('--model', required=True, default=None,
                        choices=['densenet121', 'densenet161', 'densenet169',
                                 'densenet201', 'densenet232', 'densenet264'])
    parser.add_argument('--device_target', type=str,
                        default="GPU", choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--data_url', required=True,
                        default=None, help="Location of data")
    parser.add_argument('--epoch_size', type=int,
                        default=300, help='Train epoch size.')
    parser.add_argument('--pretrained', type=bool,
                        default=False, help='Load pretrained model.')
    parser.add_argument('--keep_checkpoint_max', type=int,
                        default=10, help='Max number of checkpoint files.')
    parser.add_argument('--ckpt_save_dir', type=str,
                        default="./densenet", help='Location of training outputs.')
    parser.add_argument('--num_parallel_workers', type=int,
                        default=1, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='Number of batch size')
    parser.add_argument('--repeat_num', type=int, default=1,
                        help='Number of repetition')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='Number of classification classes')
    parser.add_argument("--lr_decay_mode", type=str,
                        default='piecewise_constant_lr', help='Learning rate decay mode.')
    # cosine decay lr
    parser.add_argument('--min_lr', type=float, default=0.0,
                        help='The end learning rate')
    parser.add_argument('--max_lr', type=float, default=0.1,
                        help='the max learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=90, help='Number of decay epochs')
    # piecewise constant lr
    parser.add_argument('--milestone', type=list,
                        default=[30, 60, 90], help='A list of milestone.')
    parser.add_argument('--learning_rate', type=list,
                        default=[0.1, 0.01, 0.001], help='A list of learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for the moving average')

    parser.add_argument('--dataset_sink_mode', type=bool,
                        default=True, help='The dataset sink mode.')
    parser.add_argument('--run_distribute', type=bool,
                        default=False, help='Run distribute.')
    parser.add_argument('--resize', type=int, default=224,
                        help='Resize the image.')

    args = parser.parse_known_args()[0]
    densenet_train(args)
