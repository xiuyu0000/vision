#!/bin/bash
# Copyright 2021
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

if [ $# != 3 ]
then
    echo "Usage: bash test.sh [device_id] [yaml_file] [checkpoint_path]"
    exit 1
fi

current_exec_path=$(pwd)
echo ${current_exec_path}

rm -rf ${current_exec_path}/device*

ulimit -c unlimited

root=${current_exec_path}

dirname_path=$root/tools
echo ${dirname_path}

python ${dirname_path}/eval.py --config $2 --device_id $1 --checkpoint_path $3 --work_dir outputs >test.log 2>&1 &
 
