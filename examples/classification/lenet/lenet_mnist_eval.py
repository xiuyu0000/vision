# Copyright 2021
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
""" LeNet eval script. """

import argparse

from mindspore import nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.train import Model

from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet


def lenet_eval(args_opt):
    """LeNet eval."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data pipeline.
    dataset = Mnist(args_opt.data_url,
                    split="test",
                    batch_size=args_opt.batch_size,
                    resize=args_opt.resize,
                    download=args_opt.download)

    dataset_eval = dataset.run()

    # Create model.
    network = lenet(args_opt.num_classes, pretrained=args_opt.pretrained)

    # Load checkpoint file for ST test.
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

    # Begin to eval
    result = model.eval(dataset_eval)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LeNet eval.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--download', type=bool, default=False, help='Download Mnist val dataset.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--ckpt_file', type=str, default=None, help='Path of the check point file.')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of batch size.')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classification.')
    parser.add_argument('--resize', type=int, default=32, help='Resize the height and weight of picture.')

    args = parser.parse_known_args()[0]
    lenet_eval(args)
