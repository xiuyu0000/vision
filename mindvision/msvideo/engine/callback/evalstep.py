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
"""print eval step"""

from mindspore.train.callback import Callback


class PrintEvalStep(Callback):
    """ print eval step """
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        print("eval: {}/{}".format(cb_params.cur_step_num, cb_params.batch_num))
