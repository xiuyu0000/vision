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
"""The module is used to register MindSpore builtin neck APIs."""

import inspect

from mindspore import nn

from mindvision.engine.class_factory import ClassFactory, ModuleType


def register_builtin_transforms():
    """ register MindSpore builtin dataset class. """
    module_names = ["Flatten"]

    for module_name in module_names:
        if not module_name.startswith('__'):
            neck = getattr(nn, module_name, None)
            if inspect.isclass(neck):
                ClassFactory.register_cls(neck, ModuleType.NECK)
