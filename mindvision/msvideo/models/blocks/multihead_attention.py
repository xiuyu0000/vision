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
"""vistr multihead_attention"""
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import numpy as mnp
from mindspore import ops
from mindspore.common.initializer import initializer, HeUniform


def linear(input_arr, weight, bias=None):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Args:
        Input: :math:`(N, *, in_features)` N is the batch size, `*` means any number of
          additional dimensions
        Weight: :math:`(out_features, in_features)`
        Bias: :math:`(out_features)`
        Output: :math:`(N, *, out_features)`
    Returns:
        tensor
    """
    if input_arr.ndim == 2 and bias is not None:
        # fused op is marginally faster
        ret = ops.BatchMatMul()(input_arr, weight.T) + bias
    else:
        output = mnp.matmul(input_arr, weight.T)
        if bias is not None:
            output += bias
        ret = output
    return ret


class MultiheadAttention(nn.Cell):
    r"""multi head attention
    Args:
        embed_dim(int): total dimension of the model
        num_heads(int): parallel attention heads
        dropout(float): a Dropout layer on attn_output_weights.Default=0.
    Inputs:
        - query(Tensor): math:`(L, N, E)` where L is the target sequence
            length, N is the batch size, E is the embedding dimension.
        - key(Tensor) : math:`(S, N, E)`, where S is the source sequence
            length, N is the batch size, E is the embedding dimension.
        - value(Tensor) : math:`(S, N, E)` where S is the source sequence
            length, N is the batch size, E is the embedding dimension.
        - key_padding_mask(Tensor):if provided, specified padding elements in
            the key will be ignored by the attention. This is an binary mask.
            When the value is True,the corresponding value on the attention
            layer will be filled with -inf.
    Outputs:
        Tensor, attn_output:(L, N, E), where L is the target sequence length,
        N is the batch size, E is the embedding dimension.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.q_in_proj_weight = Parameter(initializer('xavier_uniform',
                                                      [embed_dim, embed_dim],
                                                      mstype.float32))
        self.k_in_proj_weight = Parameter(initializer('xavier_uniform',
                                                      [embed_dim, embed_dim],
                                                      mstype.float32))
        self.v_in_proj_weight = Parameter(initializer('xavier_uniform',
                                                      [embed_dim, embed_dim],
                                                      mstype.float32))

        self.q_in_proj_bias = Parameter(initializer('zeros',
                                                    [embed_dim],
                                                    mstype.float32))
        self.k_in_proj_bias = Parameter(initializer('zeros',
                                                    [embed_dim],
                                                    mstype.float32))
        self.v_in_proj_bias = Parameter(initializer('zeros',
                                                    [embed_dim],
                                                    mstype.float32))

        self.out_proj = nn.Dense(embed_dim, embed_dim,
                                 weight_init=HeUniform())
        self.drop = nn.Dropout(1 - dropout)

    def construct(self,
                  query: Tensor,
                  key: Tensor,
                  value: Tensor,
                  key_padding_mask: Tensor):
        """construct MultiheadAttention"""
        tgt_len, bsz, embed_dim = query.shape
        scaling = self.head_dim ** -0.5

        q = linear(query, self.q_in_proj_weight, self.q_in_proj_bias)
        k = linear(key, self.k_in_proj_weight, self.k_in_proj_bias)
        v = linear(value, self.v_in_proj_weight, self.v_in_proj_bias)

        q = q * scaling

        q = q.view(tgt_len, bsz * self.num_heads,
                   self.head_dim).transpose(1, 0, 2)
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)

        src_len = k.shape[1]

        attn_output_weights = ops.BatchMatMul()(q, k.transpose(0, 2, 1))

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            key_padding_mask = ops.Tile()(
                ops.ExpandDims()(ops.ExpandDims()(key_padding_mask, 1), 2),
                (1, self.num_heads, tgt_len, 1)
            )
            attn_output_weights = attn_output_weights - key_padding_mask * \
                10000.
            attn_output_weights = attn_output_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = ops.Softmax(axis=-1)(attn_output_weights)
        attn_output_weights = self.drop(attn_output_weights)

        attn_output = ops.BatchMatMul()(attn_output_weights, v)
        attn_output = attn_output.transpose(
            1, 0, 2).view(tgt_len, bsz, embed_dim)
        attn_output = linear(
            attn_output, self.out_proj.weight, self.out_proj.bias)
        return attn_output
