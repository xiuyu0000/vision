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
"""
Init
"""
from . import convnext, efficientnet, lenet, mobilenet_v1, mobilenet_v2, mobilenet_v3, resnet, vit, vgg, googlenet, squeezenet, alexnet, densenet
from .convnext import *
from .efficientnet import *
from .lenet import *
from .mobilenet_v2 import *
from .resnet import *
from .vit import *
from .mobilenet_v1 import *
from .mobilenet_v3 import *
from .vgg import *
from .densenet import *
from .googlenet import *
from .squeezenet import *
from .alexnet import *

__all__ = []
__all__.extend(convnext.__all__)
__all__.extend(efficientnet.__all__)
__all__.extend(lenet.__all__)
__all__.extend(mobilenet_v2.__all__)
__all__.extend(resnet.__all__)
__all__.extend(vit.__all__)
__all__.extend(mobilenet_v1.__all__)
__all__.extend(mobilenet_v3.__all__)
__all__.extend(vgg.__all__)
__all__.extend(densenet.__all__)
__all__.extend(googlenet.__all__)
__all__.extend(squeezenet.__all__)
__all__.extend(alexnet.__all__)
