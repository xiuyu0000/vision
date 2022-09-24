# Copyright 2022
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
""" ViT training script. """

import argparse

from mindspore import context, load_checkpoint, load_param_into_net
from mindspore import nn

from mindspore.common import set_seed
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train import Model

from mindvision.msvideo.dataset import Kinetic400
from mindvision.msvideo.models import swin3d_t, swin3d_s, swin3d_b, swin3d_l

set_seed(42)


def swin_tiny_eval(args_opt):
    """Swin3d train."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)

    # Data Pipeline.
    dataset_eval = Kinetic400(path=args_opt.dataset_path,
                              batch_size=args_opt.batch_size,
                              split='test',
                              shuffle=True,
                              seq=32,
                              num_parallel_workers=args_opt.num_parallel_workers,
                              seq_mode='part',
                              )
    dataset_eval.transform = dataset_eval.default_transform()
    dataset_eval = dataset_eval.run()

    # Create model.
    if args_opt.model_name == "swin3d_t":
        network = swin3d_t()
    elif args_opt.model_name == "swin3d_s":
        network = swin3d_s()
    elif args_opt.model_name == "swin3d_b":
        network = swin3d_b()
    elif args_opt.model_name == "swin3d_l":
        network = swin3d_l()
    if args_opt.pretrained:
        param_dict = load_checkpoint(args_opt.pretrained_model_dir)
        load_param_into_net(network, param_dict)

    # Define loss function.
    network_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Define eval metrics.
    eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                    'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}

    # Init the model.
    model = Model(network, loss_fn=network_loss, metrics=eval_metrics)

    # Begin to train.
    print('[Start eval `{}`]'.format(args_opt.model_name))
    result = model.eval(dataset_eval, dataset_sink_mode=args_opt.dataset_sink_mode)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swin3d_t Eval.")
    parser.add_argument("--dataset_path", type=str, default="/usr/publicfile/kinetics-400")
    parser.add_argument("--model_name", type=str, default="swin3d_t")
    parser.add_argument("--device_target", type=str, default="GPU")
    parser.add_argument("--device_id", type=int, default=2)
    parser.add_argument("--pretrained", type=bool, default=True, help="Load pretrained model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of batch size.")
    parser.add_argument("--pretrained_model_dir", type=str,
                        default="./pretrained/ms_swin_tiny_patch244_window877_kinetics400_1k.ckpt",
                        help="Location of Pretrained Model.")
    parser.add_argument("--dataset_sink_mode", default=False, help="The dataset sink mode.")
    parser.add_argument("--num_parallel_workers", type=int, default=4, help="Number of parallel workers.")

    args = parser.parse_known_args()[0]

    swin_tiny_eval(args)
