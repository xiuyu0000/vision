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
""" ConvNext eval script. """

import argparse

from mindspore import nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.train import Model

from mindvision.classification.dataset import ImageNet


def convnext_eval(args_opt):
    """ ConvNext eval."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)

    # Data pipeline.
    dataset_path = args_opt.data_url

    dataset = ImageNet(dataset_path,
                       split="val",
                       num_parallel_workers=args_opt.num_parallel_workers,
                       resize=args_opt.resize,
                       batch_size=args_opt.batch_size)

    dataset_eval = dataset.run()

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

    # Load checkpoint file.
    if args_opt.ckpt_file:
        param_dict = load_checkpoint(args_opt.ckpt_file)
        load_param_into_net(network, param_dict)

    # Define loss function.
    network_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Define eval metrics.
    eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                    'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}

    # Init the model.
    model = Model(network, network_loss, metrics=eval_metrics)

    # Begin to eval.
    result = model.eval(dataset_eval)
    print(result)


def parse_args():
    """eval args"""
    parser = argparse.ArgumentParser(description='ConvNext eval.')
    parser.add_argument('--model', required=True, default=None, choices=["convnext_tiny", "convnext_small",
                                                                         "convnext_base", "convnext_large",
                                                                         "convnext_xlarge"])
    parser.add_argument('--ckpt_file', type=str, default=None, help='Path of the check point file.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--device_id', type=int, default=0, help='The machine number on which the code is run')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--num_parallel_workers', type=int, default=8, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of batch size.')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classification.')
    parser.add_argument('--resize', type=int, default=224, help='Resize the image.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convnext_eval(args)
