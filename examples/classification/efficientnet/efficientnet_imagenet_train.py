# Copyright 2022
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
""" EfficientNet training script. """

import argparse

from mindspore import nn
from mindspore import context
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from mindvision.classification.dataset import ImageNet
from mindvision.engine.loss import CrossEntropySmooth
from mindvision.engine.callback import LossMonitor

set_seed(1)


def efficientnet_train(args_opt):
    """EfficientNet train."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data Pipeline.
    if args_opt.run_distribute:
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

    # Create model.
    if args_opt.model == 'efficientnet_b0':
        from mindvision.classification.models import efficientnet_b0
        network = efficientnet_b0(args_opt.num_classes, pretrained=args_opt.pretrained, is_training=True)
    elif args_opt.model == 'efficientnet_b1':
        from mindvision.classification.models import efficientnet_b1
        network = efficientnet_b1(args_opt.num_classes, pretrained=args_opt.pretrained, is_training=True)
    elif args_opt.model == 'efficientnet_b2':
        from mindvision.classification.models import efficientnet_b2
        network = efficientnet_b2(args_opt.num_classes, pretrained=args_opt.pretrained, is_training=True)
    elif args_opt.model == 'efficientnet_b3':
        from mindvision.classification.models import efficientnet_b3
        network = efficientnet_b3(args_opt.num_classes, pretrained=args_opt.pretrained, is_training=True)
    elif args_opt.model == 'efficientnet_b4':
        from mindvision.classification.models import efficientnet_b4
        network = efficientnet_b4(args_opt.num_classes, pretrained=args_opt.pretrained, is_training=True)
    elif args_opt.model == 'efficientnet_b5':
        from mindvision.classification.models import efficientnet_b5
        network = efficientnet_b5(args_opt.num_classes, pretrained=args_opt.pretrained, is_training=True)
    elif args_opt.model == 'efficientnet_b6':
        from mindvision.classification.models import efficientnet_b6
        network = efficientnet_b6(args_opt.num_classes, pretrained=args_opt.pretrained, is_training=True)
    elif args_opt.model == 'efficientnet_b7':
        from mindvision.classification.models import efficientnet_b7
        network = efficientnet_b7(args_opt.num_classes, pretrained=args_opt.pretrained, is_training=True)

    network.set_train(True)

    # Set lr scheduler.
    lr = nn.cosine_decay_lr(min_lr=args_opt.min_lr, max_lr=args_opt.max_lr,
                            total_step=args_opt.epoch_size * step_size, step_per_epoch=step_size,
                            decay_epoch=args_opt.decay_epoch)

    # Define optimizer.

    network_opt = nn.RMSProp(network.trainable_params(),
                             learning_rate=lr,
                             momentum=args_opt.momentum,
                             )

    # Define loss function.
    network_loss = CrossEntropySmooth(
        sparse=True, reduction="mean", smooth_factor=args_opt.smooth_factor, classes_num=args_opt.num_classes
    )

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=step_size, keep_checkpoint_max=args_opt.keep_checkpoint_max
    )
    ckpt_callback = ModelCheckpoint(prefix='efficientnet', directory=ckpt_save_dir, config=ckpt_config)
    # Init the model.
    model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics={'acc'})

    # Begin to train.
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor(lr)],
                dataset_sink_mode=args_opt.dataset_sink_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Efficientnet train.')
    parser.add_argument('--model', required=True, default=None,
                        choices=["efficientnet_b0",
                                 "efficientnet_b1",
                                 "efficientnet_b2",
                                 "efficientnet_b3",
                                 "efficientnet_b4",
                                 "efficientnet_b5",
                                 "efficientnet_b6",
                                 "efficientnet_b7"]
                        )
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--epoch_size', type=int, default=150, help='Train epoch size.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='Max number of checkpoint files.')
    parser.add_argument('--ckpt_save_dir', type=str, default="./efficientnet", help='Location of training outputs.')
    parser.add_argument('--num_parallel_workers', type=int, default=8, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size.')
    parser.add_argument('--repeat_num', type=int, default=1, help='Number of repeat.')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classification.')
    parser.add_argument('--min_lr', type=float, default=0.0, help='The end learning rate.')
    parser.add_argument('--max_lr', type=float, default=0.1, help='The max learning rate.')
    parser.add_argument('--decay_epoch', type=int, default=90, help='Number of decay epochs.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the moving average.')
    parser.add_argument('--smooth_factor', type=float, default=0.1, help='The smooth factor.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='The dataset sink mode.')
    parser.add_argument('--run_distribute', type=bool, default=True, help='Run distribute.')
    parser.add_argument('--resize', type=int, default=224, help='Resize the image.')

    args = parser.parse_known_args()[0]

    efficientnet_train(args)
