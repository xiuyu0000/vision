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
"""head for VisTR"""

from mindspore import nn
from mindspore import ops
from mindspore.common import initializer as msinit
from mindspore.common.initializer import initializer, HeUniform

from mindvision.engine.class_factory import ClassFactory, ModuleType
from mindvision.msvideo.models.blocks.deform_conv import BatchMapOffsets


@ClassFactory.register(ModuleType.HEAD)
class VisTRHead(nn.Cell):
    """VisTRHead:Simple convolutional head, using group norm.
        Upsampling is done using a FPN approach

    Args:
        dim(int):Size of the embeddings (dimension of the transformer) +
                Number of attention heads inside the transformer's attentions
        fpn_dims(dict):three dims for FPN
        context_dim(int):Size of the embeddings (dimension of the transformer)

    Returns:
        Tensor
    """

    def __init__(self, dim, fpn_dims, context_dim):
        """initialize convolution layers and groupNormalization layers"""
        super(VisTRHead, self).__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4,
                      context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = nn.Conv2d(
            dim, dim, 3, padding=1,
            pad_mode='pad', has_bias=True, weight_init='normal')
        self.gn1 = nn.GroupNorm(8, dim)
        self.lay2 = nn.Conv2d(
            dim, inter_dims[1], 3, padding=1,
            pad_mode='pad', has_bias=True, weight_init='normal')
        self.gn2 = nn.GroupNorm(8, inter_dims[1])
        self.lay3 = nn.Conv2d(
            inter_dims[1], inter_dims[2], 3, padding=1,
            pad_mode='pad', has_bias=True, weight_init='normal')
        self.gn3 = nn.GroupNorm(8, inter_dims[2])
        self.lay4 = nn.Conv2d(
            inter_dims[2], inter_dims[3], 3, padding=1,
            pad_mode='pad', has_bias=True, weight_init='normal')
        self.gn4 = nn.GroupNorm(8, inter_dims[3])
        self.lay5 = nn.Conv2d(
            inter_dims[3], inter_dims[4], 3, padding=1,
            pad_mode='pad', has_bias=True, weight_init='normal')
        self.gn5 = nn.GroupNorm(8, inter_dims[4])
        self.conv_offset = nn.Conv2d(
            inter_dims[3], inter_dims[2], 3,
            has_bias=True, weight_init='zeros')
        self.relu = ops.ReLU()
        self.dim = dim
        self.batch_map_offsets = BatchMapOffsets()

        self.adapter1 = nn.Conv2d(
            fpn_dims[0], inter_dims[1], 1, has_bias=True, weight_init='normal')
        self.adapter2 = nn.Conv2d(
            fpn_dims[1], inter_dims[2], 1, has_bias=True, weight_init='normal')
        self.adapter3 = nn.Conv2d(
            fpn_dims[2], inter_dims[3], 1, has_bias=True, weight_init='normal')

        self.reshape = ops.Reshape()
        self.expand_dim = ops.ExpandDims()

        self.concat = ops.Concat(axis=1)
        for name, m in self.cells_and_names():
            if name == "conv_offset":
                msinit.Constant(0)(m.weight)
                msinit.Constant(0)(m.bias)
            else:
                if isinstance(m, nn.Conv2d):
                    m.weight = initializer(
                        HeUniform(negative_slope=1), m.weight.shape)
                    msinit.Constant(0)(m.bias)

    def construct(self, x, bbox_mask, fpns):
        """
        construct vistr head

        Args:
            x(Tensor):sequence of encoded features
            bbox_mask(Tensor): the attention softmax of bbox
            fpns(list[Tensor]):images features without positional encoding
        """
        mask_shape = bbox_mask.shape
        mask_shape = (-1, mask_shape[2], mask_shape[3], mask_shape[4])
        bbox_mask_flatten = bbox_mask.reshape(mask_shape)
        expand_x = self._expand(x, bbox_mask.shape[1])
        x = self.concat((expand_x, bbox_mask_flatten))
        x = self.lay1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = self.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = self._expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        interpolate = ops.ResizeNearestNeighbor(size=cur_fpn.shape[-2:])
        x = cur_fpn + interpolate(x)
        x = self.lay3(x)
        x = self.gn3(x)
        x = self.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = self._expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        interpolate = ops.ResizeNearestNeighbor(size=cur_fpn.shape[-2:])
        x = cur_fpn + interpolate(x)
        x = self.lay4(x)
        x = self.gn4(x)
        x = self.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = self._expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        interpolate = ops.ResizeNearestNeighbor(size=cur_fpn.shape[-2:])
        x = cur_fpn + interpolate(x)

        # dcn for the last layer
        x_shape = x.shape
        offset = self.conv_offset(x)
        offset = self._to_bc_h_w_2(offset, x_shape)
        x = self._to_bc_h_w(x, x_shape)
        x = self.batch_map_offsets(x, offset)
        x = self._to_b_h_w_c(x, x_shape)
        x = self.lay5(x)
        x = self.gn5(x)
        x = self.relu(x)
        return x

    def _to_bc_h_w_2(self, x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = self.reshape(x, (-1, x_shape[2], x_shape[3], 2))
        return x

    def _to_bc_h_w(self, x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = self.reshape(x, (-1, x_shape[2], x_shape[3]))
        return x

    def _to_b_h_w_c(self, x, x_shape):
        """(b*c, h*w) -> (b, c, h, w)"""
        x = self.reshape(
            x, (x_shape[0], x_shape[1],
                x_shape[2], x_shape[3])
        )
        return x

    def _expand(self, tensor, length):
        """expand tensor"""
        tensor = self.expand_dim(tensor, 1)
        tensor = ops.tile(tensor, (1, length, 1, 1, 1))
        dim0 = tensor.shape[0]
        dim1 = tensor.shape[1]
        dim2 = tensor.shape[2]
        dim3 = tensor.shape[3]
        dim4 = tensor.shape[4]
        tensor = tensor.resize(dim0 * dim1, dim2, dim3, dim4)
        return tensor
