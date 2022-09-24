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
""" SSD eval script. """

import argparse
import os

from mindspore import context

from mindvision.msdetection.dataset import COCODetection
from mindvision.msdetection.models.ssd import ssd_mobilenet_v2
from mindvision.msdetection.models.detection_engine import SSDDetectionEngine
from mindvision.msdetection.models.utils.ssd_utils import apply_eval
from mindvision.msdetection.dataset.transforms import DetectionDecode, DetectionResize, DetectionNormalize, \
    DetectionHWC2CHW


def ssd_eval(args_opt):
    """ssd train."""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)

    # Data Pipeline.
    transforms = [
        DetectionDecode(),
        DetectionResize(args_opt.resize),
        DetectionNormalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                           std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
        DetectionHWC2CHW()
    ]
    dataset = COCODetection(args_opt.data_url,
                            split="val",
                            transforms=transforms,
                            batch_size=args_opt.batch_size,
                            num_parallel_workers=args_opt.num_parallel_workers,
                            remove_invalid_annotations=True,
                            filter_crowd_annotations=True,
                            trans_record=True)

    dataset_eval = dataset.run()

    # Create model.
    if args_opt.backbone == "mobilenet_v2":
        network = ssd_mobilenet_v2(args_opt.num_classes, pretrained=args_opt.pretrained)

    network.set_train(False)

    # create detection engine.
    ann_file = os.path.join(args_opt.data_url, "annotations", "instances_val2017.json")
    detection_engine = SSDDetectionEngine(num_classes=args_opt.num_classes,
                                          ann_file=ann_file,
                                          min_score=args_opt.min_score,
                                          nms_threshold=args_opt.nms_threshold,
                                          max_boxes=args_opt.max_boxes)

    apply_eval(network, dataset_eval, detection_engine)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSD eval.')
    parser.add_argument('--backbone', required=True, default=None,
                        choices=["mobilenet_v2"])
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--pretrained', type=bool, default=True, help='Load pretrained model.')
    parser.add_argument('--num_parallel_workers', type=int, default=8, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size.')
    parser.add_argument('--num_classes', type=int, default=81, help='Number of classification.')
    parser.add_argument('--min_score', type=float, default=0.1, help='Minimum score.')
    parser.add_argument('--nms_threshold', type=float, default=0.6, help='Nms threshold.')
    parser.add_argument('--max_boxes', type=int, default=100, help='Maximum boxes.')
    parser.add_argument('--resize', type=tuple, default=(300, 300), help='Resize the height and weight of picture.')

    args = parser.parse_known_args()[0]
    ssd_eval(args)
