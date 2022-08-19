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
"""X3D eval script."""

import argparse

from mindspore import nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from mindvision.msvideo.dataset import Kinetic400
from mindvision.msvideo.dataset.transforms import VideoRandomCrop, VideoRandomHorizontalFlip
from mindvision.msvideo.dataset.transforms import VideoResize, VideoToTensor
from mindvision.msvideo.models.x3d import x3d_m, x3d_l, x3d_s, x3d_xs
from mindvision.msvideo.engine.callback import PrintEvalStep


def x3d_kinetics400_eval(args_opt):
    """X3D eval"""
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target)

    # Data Pipeline.
    dataset_eval = Kinetic400(args_opt.data_url,
                              split="val",
                              seq=args_opt.seq,
                              num_parallel_workers=args_opt.num_parallel_workers,
                              shuffle=True,
                              batch_size=args_opt.batch_size,
                              repeat_num=args_opt.repeat_num)

    # perpare dataset.
    transforms = [VideoResize([256, 256]),
                  VideoRandomCrop([224, 224]),
                  VideoRandomHorizontalFlip(0.5),
                  VideoToTensor()]
    dataset_eval.transform = transforms
    dataset_eval = dataset_eval.run()

    # Create model.
    if args_opt.model_name == "x3d_m":
        network = x3d_m(num_classes=args_opt.num_classes)
    elif args_opt.model_name == "x3d_s":
        network = x3d_s(num_classes=args_opt.num_classes)
    elif args_opt.model_name == "x3d_xs":
        network = x3d_xs(num_classes=args_opt.num_classes)
    elif args_opt.model_name == "x3d_l":
        network = x3d_l(num_classes=args_opt.num_classes)

    # Define loss function.
    network_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Load pretrained model.
    if args_opt.pretrained:
        param_dict = load_checkpoint(args_opt.pretrained_model)
        load_param_into_net(network, param_dict)

    # Define eval_metrics.
    eval_metrics = {'Loss': nn.Loss(),
                    'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                    'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}
    print_cb = PrintEvalStep()

    # Init the model.
    model = Model(network, loss_fn=network_loss, metrics=eval_metrics)

    # Begin to eval.
    print('[Start eval `{}`]'.format('x3d_kinetics400'))
    result = model.eval(dataset_eval,
                        callbacks=[print_cb],
                        dataset_sink_mode=args_opt.dataset_sink_mode)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='X3D eval.')
    parser.add_argument('--device_target', type=str, default="GPU",
                        choices=["Ascend", "GPU", "CPU"])
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
    parser.add_argument("--model_name", type=str, default="x3d_m",
                        help="Name of model.", choices=["x3d_m", "x3d_l", "x3d_s", "x3d_xs"])
    parser.add_argument('--num_classes', type=int, default=400,
                        help='Number of classification.')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Load pretrained model.')
    parser.add_argument('--pretrained_model', type=str, default="",
                        help='Location of Pretrained Model.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False,
                        help='The dataset sink mode.')

    args = parser.parse_known_args()[0]
    x3d_kinetics400_eval(args)
