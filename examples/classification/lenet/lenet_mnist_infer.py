# Copyright 2021 Huawei Technologies Co., Ltd
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
""" LeNet infer script."""

import argparse
import numpy as np

from mindspore import context, Tensor
from mindspore.train import Model

from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet


def lenet_infer(args_opt):
    """LeNet infer."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data pipeline.
    dataset = Mnist(args_opt.data_url,
                    split="infer",
                    batch_size=args_opt.batch_size,
                    resize=args_opt.resize)

    dataset_infer = dataset.run()

    # Create model.
    network = lenet(args_opt.num_classes, pretrained=args_opt.pretrained, ckpt_file=args_opt.ckpt_file)

    # Init the model.
    model = Model(network)

    # Begin to infer
    for data in dataset_infer.create_dict_iterator(output_numpy=True):
        image = data["image"]
        image = Tensor(np.expand_dims(image, axis=1))
        prob = model.predict(image)
        label = np.argmax(prob.asnumpy(), axis=1)
        for i in label:
            predict = dataset.index2label[i]
            output = {i: predict}
            print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LeNet infer.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--ckpt_file', type=str, default=None, help='Path of the check point file.')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size.')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classification.')
    parser.add_argument('--resize', type=int, default=32, help='Resize the height and weight of picture.')

    args = parser.parse_known_args()[0]
    lenet_infer(args)
