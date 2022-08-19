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
""" DenseNet infer script. """

import argparse

import numpy as np

from mindspore import context, Tensor
from mindspore.train import Model

from mindvision.classification.dataset import ImageNet
from mindvision.dataset.download import read_dataset
from mindvision.classification.utils.image import show_result


def densenet_infer(args_opt):
    """ResNet infer."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data pipeline.
    dataset = ImageNet(args_opt.data_url,
                       split="infer",
                       num_parallel_workers=args_opt.num_parallel_workers,
                       resize=args_opt.resize,
                       batch_size=args_opt.batch_size)

    dataset_infer = dataset.run()

    # Create model.
    if args_opt.model == 'densenet121':
        from mindvision.classification.models import densenet121
        network = densenet121(args_opt.num_classes, pretrained=args_opt.pretrained)
    elif args_opt.model == 'densenet161':
        from mindvision.classification.models import densenet161
        network = densenet161(args_opt.num_classes, pretrained=args_opt.pretrained)
    elif args_opt.model == 'densenet169':
        from mindvision.classification.models import densenet169
        network = densenet169(args_opt.num_classes, pretrained=args_opt.pretrained)
    elif args_opt.model == 'densenet201':
        from mindvision.classification.models import densenet201
        network = densenet201(args_opt.num_classes, pretrained=args_opt.pretrained)
    elif args_opt.model == 'densenet232':
        from mindvision.classification.models import densenet232
        network = densenet232(args_opt.num_classes, pretrained=args_opt.pretrained)
    elif args_opt.model == 'densenet264':
        from mindvision.classification.models import densenet264
        network = densenet264(args_opt.num_classes, pretrained=args_opt.pretrained)

    # Init the model.
    model = Model(network)

    # Begin to infer
    image_list, _ = read_dataset(args_opt.data_url)
    for data in dataset_infer.create_dict_iterator(output_numpy=True):
        image = data["image"]
        image = Tensor(image)
        prob = model.predict(image)
        label = np.argmax(prob.asnumpy(), axis=1)
        for i, v in enumerate(label):
            predict = dataset.index2label[v]
            output = {v: predict}
            print(output)
            show_result(img=image_list[i], result=output, out_file=image_list[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet infer.')
    parser.add_argument('--model', required=True, default=None,
                        choices=['densenet121', 'densenet161', 'densenet169',
                                 'densenet201', 'densenet232', 'densenet264'])
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', required=True, default=None,
                        help='Root of infer data and ILSVRC2012_devkit_t12.tar.gz.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--num_parallel_workers', type=int, default=8, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size.')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classification.')
    parser.add_argument('--resize', type=int, default=224, help='Resize the height and weight of picture.')

    args = parser.parse_known_args()[0]
    densenet_infer(args)
