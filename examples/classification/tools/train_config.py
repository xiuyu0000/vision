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
"""MindSpore Vision Classification uses yaml file for training."""

from mindspore.common import set_seed

from mindvision.engine.utils.config import parse_args, Config
from mindvision.classification.tools.train import train_config

set_seed(1)

if __name__ == '__main__':
    args = parse_args()
    cfg = Config(args.config)
    if cfg.get("image_size"):
        image_size = cfg.image_size
        cfg.data_loader.train.map.operations[0]["size"] = image_size
        if cfg.model.backbone["image_size"]:
            cfg.model.backbone["image_size"] = image_size[0]
    train_config(cfg)
