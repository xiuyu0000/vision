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
"""Video Pooling neck."""

from typing import Union, List, Tuple

from mindspore import nn
from mindspore import ops

from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.msvideo.engine.ops import AvgPool3D, AdaptiveAvgPool3D, MaxPool3D


@ClassFactory.register(ModuleType.NECK)
class GlobalAvgPooling3D(nn.Cell):
    """
    A module of Global average pooling for 3D video features.

    Args:
        keep_dims (bool): Specifies whether to keep dimension shape the same as input feature.
            E.g. `True`. Default: False

    Returns:
        Tensor, output tensor.
    """

    def __init__(self,
                 keep_dims: bool = True
                 ) -> None:
        super(GlobalAvgPooling3D, self).__init__()
        self.mean = ops.ReduceMean(keep_dims=keep_dims)

    def construct(self, x):
        x = self.mean(x, (2, 3, 4))
        return x


@ClassFactory.register(ModuleType.NECK)
class AvgPooling3D(nn.Cell):
    """
    A module of average pooling for 3D video features.

    Args:
        kernel_size(Union[int, List[int], Tuple[int]]): The size of kernel window used to take the
            average value, Default: (1, 1, 1).
        strides(Union[int, List[int], Tuple[int]]): The distance of kernel moving. Default: (1, 1, 1).

    Inputs:
        x(Tensor): The input Tensor.

    Returns:
        Tensor, the pooled Tensor.
    """

    def __init__(self,
                 kernel_size: Union[int, List[int], Tuple[int]] = (1, 1, 1),
                 strides: Union[int, List[int], Tuple[int]] = (1, 1, 1),
                 ) -> None:
        super(AvgPooling3D, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        kernel_size = tuple(kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides, strides)
        strides = tuple(strides)

        self.pool = AvgPool3D(kernel_size, strides)

    def construct(self, x):
        x = self.pool(x)
        return x


@ClassFactory.register(ModuleType.NECK)
class AdaptiveAvgPooling3D(nn.Cell):
    """Applies a 3D adaptive average pooling over an input tensor which is typically of shape
        :math:`(N, C, D_{in}, H_{in}, W_{in})` and output shape
        :math:`(N, C, D_{out}, H_{out}, W_{out})`. where :math:`N` is batch size. :math:`C` is
        channel number.

    Args:
        output_size(Union[int, List[int], Tuple[int]]): The target output size of the form D x H x W.
            Can be a tuple (D, H, W) or a single number D for a cube D x D x D.

    Inputs:
        x(Tensor): The input Tensor in the form of :math:`(N, C, D_{in}, H_{in}, W_{in})`.

    Returns:
        Tensor, the pooled Tensor in the form of :math:`(N, C, D_{out}, H_{out}, W_{out})`.
    """

    def __init__(self,
                 output_size: Union[int, List[int], Tuple[int]] = 1,
                 ) -> None:
        super(AdaptiveAvgPooling3D, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)
        output_size = tuple(output_size)

        self.pool = AdaptiveAvgPool3D(output_size)

    def construct(self, x):
        x = self.pool(x)
        return x


@ClassFactory.register(ModuleType.NECK)
class MaxPooling3D(nn.Cell):
    """
    A module of max pooling for 3D video features.

    Args:
        kernel_size(Union[int, List[int], Tuple[int]]): The size of kernel window used to take the
            average value, Default: (1, 1, 1).
        strides(Union[int, List[int], Tuple[int]]): The distance of kernel moving. Default: (1, 1, 1).

    Inputs:
        x(Tensor): The input Tensor.

    Returns:
        Tensor, the pooled Tensor.
    """

    def __init__(self,
                 kernel_size: Union[int, List[int], Tuple[int]] = (1, 1, 1),
                 strides: Union[int, List[int], Tuple[int]] = (1, 1, 1),
                 ) -> None:
        super(MaxPooling3D, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        kernel_size = tuple(kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides, strides)
        strides = tuple(strides)

        self.pool = MaxPool3D(kernel_size, strides)

    def construct(self, x):
        x = self.pool(x)
        return x
