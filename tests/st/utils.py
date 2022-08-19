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

import os
import re


def parse_log_file(pattern, log_path):
    value_list = []
    with open(log_path, "r") as file:
        for line in file.readlines():
            match_result = re.search(pattern, line)
            if match_result is not None:
                value_list.append(float(match_result.group(1)))
    if not value_list:
        print("pattern is", pattern)
        cmd = "cat {}".format(log_path)
        os.system(cmd)
    return value_list
