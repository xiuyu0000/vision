#!/bin/bash
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

CURPATH="$(dirname "$0")"
. ${CURPATH}/cache_util.sh

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8

rm -rf ./train_parallel
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp *.sh ./train_parallel
cp -r ../src ./train_parallel
cd ./train_parallel || exit


mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
       python train.py -c config/resnet/resnet18_imagenet2012.yaml
