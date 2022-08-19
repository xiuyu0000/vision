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
""" MobileNetV2 infer script. """

import argparse

import numpy as np

from mindspore import context, Tensor
from mindspore.train import Model

from mindvision.classification.models import mobilenet_v2
from mindvision.classification.dataset import ImageNet
from mindvision.classification.utils.image import show_result
from mindvision.dataset.download import read_dataset


def mobilenet_v2_infer(args_opt):
    """MobileNetV2 infer."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data pipeline.
    dataset = ImageNet(args_opt.data_url,
                       split="infer",
                       num_parallel_workers=args_opt.num_parallel_workers,
                       resize=args_opt.resize,
                       batch_size=args_opt.batch_size)

    dataset_infer = dataset.run()

    # Create model.
    network = mobilenet_v2(args_opt.num_classes, alpha=args_opt.alpha, pretrained=args_opt.pretrained,
                           resize=args_opt.resize)

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
    parser = argparse.ArgumentParser(description='MobileNetV2 infer.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', required=True, default=None,
                        help='Root of infer data and ILSVRC2012_devkit_t12.tar.gz.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--num_parallel_workers', type=int, default=8, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size.')
    parser.add_argument('--num_classes', type=int, default=1001, help='Number of classification.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Magnification factor.')
    parser.add_argument('--resize', type=int, default=224, help='Resize the height and weight of picture.')

    args = parser.parse_known_args()[0]
    mobilenet_v2_infer(args)
