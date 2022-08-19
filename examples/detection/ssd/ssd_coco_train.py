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
""" SSD training script. """

import argparse

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from mindvision.msdetection.dataset import COCODetection
from mindvision.msdetection.models.ssd import ssd_mobilenet_v2
from mindvision.msdetection.models.utils import TrainingWrapper
from mindvision.msdetection.internals.anchor import GenerateDefaultBoxes
from mindvision.msdetection.models.utils import SSDEncoder
from mindvision.engine.callback import LossMonitor
from mindvision.msdetection.dataset.transforms import DetectionDecode, RandomSampleCrop, DetectionResize, \
    DetectionToPercentCoords, AssignGTToDefaultBox, DetectionRandomColorAdjust, DetectionNormalize, DetectionHWC2CHW

set_seed(1)


def ssd_train(args_opt):
    """ssd train."""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target,
                        enable_graph_kernel=True,
                        graph_kernel_flags="--enable_parallel_fusion --enable_expand_ops=Conv2D")

    # Data Pipeline.
    anchor_generator = GenerateDefaultBoxes(img_shape=args_opt.resize,
                                            steps=args_opt.steps,
                                            max_scale=args_opt.max_scale,
                                            min_scale=args_opt.min_scale,
                                            num_default=args_opt.num_default,
                                            feature_size=args_opt.feature_size,
                                            aspect_ratios=args_opt.aspect_ratios)
    ssd_encoder = SSDEncoder(match_threshold=args_opt.match_threshold,
                             prior_scaling=args_opt.prior_scaling,
                             anchor_generator=anchor_generator)
    transforms = [
        DetectionDecode(),
        RandomSampleCrop(),
        DetectionResize(args_opt.resize),
        DetectionToPercentCoords(),
        AssignGTToDefaultBox(ssd_encoder),
        DetectionRandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
        DetectionNormalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                           std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
        DetectionHWC2CHW()
    ]
    if args_opt.run_distribute:
        init("nccl")
        rank_id = get_rank()
        device_num = get_group_size()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

        if args_opt.all_reduce_fusion_config:
            context.set_auto_parallel_context(all_reduce_fusion_config=args_opt.all_reduce_fusion_config)

        dataset = COCODetection(args_opt.data_url,
                                split="train",
                                transforms=transforms,
                                batch_size=args_opt.batch_size,
                                repeat_num=args_opt.repeat_num,
                                num_parallel_workers=args_opt.num_parallel_workers,
                                shuffle=True,
                                num_shards=device_num,
                                shard_id=rank_id,
                                remove_invalid_annotations=True,
                                filter_crowd_annotations=True,
                                trans_record=True)
        ckpt_save_dir = args_opt.ckpt_save_dir + "_ckpt_" + str(rank_id) + "/"
    else:
        dataset = COCODetection(args_opt.data_url,
                                split="train",
                                transforms=transforms,
                                batch_size=args_opt.batch_size,
                                repeat_num=args_opt.repeat_num,
                                num_parallel_workers=args_opt.num_parallel_workers,
                                shuffle=True,
                                remove_invalid_annotations=True,
                                filter_crowd_annotations=True,
                                trans_record=True)
        ckpt_save_dir = args_opt.ckpt_save_dir

    dataset_train = dataset.run()
    step_size = dataset_train.get_dataset_size()

    # Create model.
    if args_opt.backbone == "mobilenet_v2":
        network = ssd_mobilenet_v2(args_opt.num_classes, pretrained=args_opt.pretrained)
        if args_opt.use_float16:
            network.to_float(ms.float16)

    # Set lr scheduler.
    if args_opt.lr_decay_mode == 'cosine_decay_lr':
        lr = nn.cosine_decay_lr(min_lr=args_opt.min_lr, max_lr=args_opt.max_lr,
                                total_step=args_opt.epoch_size * step_size, step_per_epoch=step_size,
                                decay_epoch=args_opt.decay_epoch)
    elif args_opt.lr_decay_mode == 'piecewise_constant_lr':
        lr = nn.piecewise_constant_lr(args_opt.milestone, args_opt.learning_rates)

    # Define optimizer.
    network_opt = nn.Momentum(network.trainable_params(), lr, args_opt.momentum,
                              args_opt.weight_decay, args_opt.loss_scale)

    # Define training wrapper.
    network = TrainingWrapper(network, network_opt, args_opt.loss_scale, args_opt.use_global_norm)

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=step_size,
        keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix=args_opt.backbone,
                                    directory=ckpt_save_dir,
                                    config=ckpt_config)
    # Init the model.
    model = Model(network)

    # Begin to train.
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor(lr)],
                dataset_sink_mode=args_opt.dataset_sink_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSD train.')
    parser.add_argument('--backbone', required=True, default=None,
                        choices=["mobilenet_v2"])
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--epoch_size', type=int, default=90, help='Train epoch size.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='Max number of checkpoint files.')
    parser.add_argument('--ckpt_save_dir', type=str, default="./ssd", help='Location of training outputs.')
    parser.add_argument('--num_parallel_workers', type=int, default=8, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size.')
    parser.add_argument('--repeat_num', type=int, default=1, help='Number of repeat.')
    parser.add_argument('--num_classes', type=int, default=81, help='Number of classification.')
    parser.add_argument('--lr_decay_mode', type=str, default="cosine_decay_lr", help='Learning rate decay mode.')
    parser.add_argument('--min_lr', type=float, default=0.0, help='The end learning rate.')
    parser.add_argument('--max_lr', type=float, default=0.1, help='The max learning rate.')
    parser.add_argument('--decay_epoch', type=int, default=90, help='Number of decay epochs.')
    parser.add_argument('--milestone', type=list, default=None, help='A list of milestone.')
    parser.add_argument('--learning_rates', type=list, default=None, help='A list of learning rates.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the moving average.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 penalty).')
    parser.add_argument('--loss_scale', type=float, default=1.0, help='A floating point value for the loss scale.')
    parser.add_argument('--use_global_norm', type=bool, default=False, help='Use global norm.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=True, help='The dataset sink mode.')
    parser.add_argument('--run_distribute', type=bool, default=True, help='Run distribute.')
    parser.add_argument('--use_float16', type=bool, default=True, help='Use data type of float16.')
    parser.add_argument('--all_reduce_fusion_config', type=list, default=None, help='Use data type of float16.')
    parser.add_argument('--resize', type=tuple, default=(300, 300), help='Resize the height and weight of picture.')
    parser.add_argument('--steps', type=list, default=[16, 32, 64, 100, 150, 300], help='steps of layers')
    parser.add_argument('--max_scale', type=float, default=0.95, help='The max scale.')
    parser.add_argument('--min_scale', type=float, default=0.2, help='The min scale.')
    parser.add_argument('--num_default', type=list, default=[3, 6, 6, 6, 6, 6], help='The number of default box.')
    parser.add_argument('--feature_size', type=list, default=[19, 10, 5, 3, 2, 1], help='The size of feature map.')
    parser.add_argument('--aspect_ratios', type=list, default=[[], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
                        help='Aspect ratios of layers.')
    parser.add_argument('--match_threshold', type=float, default=0.5, help='Match threshold.')
    parser.add_argument('--prior_scaling', type=list, default=[0.1, 0.2], help='Priority scaling value')

    args = parser.parse_known_args()[0]
    ssd_train(args)
