# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.lr_schedule.py
# org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Efficientnet eval script. """

import argparse

from mindspore import nn
from mindspore import context
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.communication import init, get_rank, get_group_size

from mindvision.classification.dataset import ImageNet
from mindvision.engine.loss import CrossEntropySmooth


def efficientnet_eval(args_opt):
    """Efficientnet eval."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data pipeline.
    dataset_path = args_opt.data_url
    if args_opt.run_distribute:
        init("nccl")
        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        dataset = ImageNet(dataset_path,
                           split="val",
                           num_parallel_workers=args_opt.num_parallel_workers,
                           resize=args_opt.resize,
                           num_shards=device_num,
                           shard_id=rank_id,
                           batch_size=args_opt.batch_size)
    else:
        dataset = ImageNet(dataset_path,
                           split="val",
                           num_parallel_workers=args_opt.num_parallel_workers,
                           resize=args_opt.resize,
                           batch_size=args_opt.batch_size)
    dataset_eval = dataset.run()

    # Create model.
    if args_opt.model == 'efficientnet_b0':
        from mindvision.classification.models import efficientnet_b0
        network = efficientnet_b0(args_opt.num_classes, pretrained=args_opt.pretrained)
    elif args_opt.model == 'efficientnet_b1':
        from mindvision.classification.models import efficientnet_b1
        network = efficientnet_b1(args_opt.num_classes, pretrained=args_opt.pretrained)
    elif args_opt.model == 'efficientnet_b2':
        from mindvision.classification.models import efficientnet_b2
        network = efficientnet_b2(args_opt.num_classes, pretrained=args_opt.pretrained)
    elif args_opt.model == 'efficientnet_b3':
        from mindvision.classification.models import efficientnet_b3
        network = efficientnet_b3(args_opt.num_classes, pretrained=args_opt.pretrained)
    elif args_opt.model == 'efficientnet_b4':
        from mindvision.classification.models import efficientnet_b4
        network = efficientnet_b4(args_opt.num_classes, pretrained=args_opt.pretrained)
    elif args_opt.model == 'efficientnet_b5':
        from mindvision.classification.models import efficientnet_b5
        network = efficientnet_b5(args_opt.num_classes, pretrained=args_opt.pretrained)
    elif args_opt.model == 'efficientnet_b6':
        from mindvision.classification.models import efficientnet_b6
        network = efficientnet_b6(args_opt.num_classes, pretrained=args_opt.pretrained)
    elif args_opt.model == 'efficientnet_b7':
        from mindvision.classification.models import efficientnet_b7
        network = efficientnet_b7(args_opt.num_classes, pretrained=args_opt.pretrained)

    network.set_train(False)

    # Define loss function.
    network_loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=args_opt.smooth_factor,
                                      classes_num=args_opt.num_classes)

    # Define eval metrics.
    eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                    'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}

    # Init the model.
    model = Model(network, network_loss, metrics=eval_metrics)

    # Begin to eval.
    result = model.eval(dataset_eval)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Efficientnet eval.')
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
    parser.add_argument('--checkpoint', type=str, default=None, help='Location of checkpoint files.')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--num_parallel_workers', type=int, default=8, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size.')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classification.')
    parser.add_argument('--smooth_factor', type=float, default=0.1, help='The smooth factor.')
    parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute.')
    parser.add_argument('--resize', type=int, default=224, help='Resize the image.')

    args = parser.parse_known_args()[0]
    efficientnet_eval(args)
