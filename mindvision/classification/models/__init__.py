# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Init classification models."""

from . import convnext, efficientnet, lenet5, mobilenetv2, resnet, googlenet, squeezenet, alexnet, vision_transform, backbones, densenet

from .builder import *
from .backbones import *
from .blocks import *
from .classifiers import *
from .head import *
from .neck import *

from .convnext import *
from .efficientnet import *
from .lenet5 import *
from .resnet import *
from .mobilenetv2 import *
from .mobilenetv3 import *
from .vision_transform import *
from .vgg import *
from .densenet import *
from .googlenet import *
from .squeezenet import *
from .alexnet import *

__all__ = []
__all__.extend(backbones.__all__)
__all__.extend(convnext.__all__)
__all__.extend(efficientnet.__all__)
__all__.extend(lenet5.__all__)
__all__.extend(mobilenetv2.__all__)
__all__.extend(mobilenetv3.__all__)
__all__.extend(resnet.__all__)
__all__.extend(vision_transform.__all__)
__all__.extend(vgg.__all__)
__all__.extend(densenet.__all__)
__all__.extend(googlenet.__all__)
__all__.extend(squeezenet.__all__)
__all__.extend(alexnet.__all__)
