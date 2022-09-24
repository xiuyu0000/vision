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

""" attention for vibe"""
import mindspore
from mindspore import ops
from mindspore import nn
from mindspore.common.initializer import initializer, Uniform


class SelfAttention(nn.Cell):
    """
    Selfattention of vibe

    Args:
        attention_size (int): Channel number of attention layer input
        layers (int):  mlp number of attention
        dropout(float): Args of Dropout
        non_linearity: activation function

    Inputs:
        x(a 3D tensor): shape(batch, len, hidden_size)

    Returns:
        scores(a 2D Tensor): shape(batch, len)
        representations: sum the hidden states
       """

    def __init__(self, attention_size,
                 batch_first=False,
                 layers=1,
                 dropout=0.0,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()
        self.batch_first = batch_first
        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            print('the' + str(i) + 'th mlp')
            modules.append(nn.Dense(attention_size, attention_size))
            modules.append(activation)
            if dropout != 0.0:
                modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Dense(attention_size, 1))
        modules.append(activation)
        if dropout != 0.0:
            modules.append(nn.Dropout(dropout))

        modules[0].weight = initializer(Uniform(0.1), shape=modules[0].weight.shape, dtype=mindspore.float32)
        modules[0].bias.data.fill(0.01)
        self.attention = nn.SequentialCell(*modules)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        """vibe attention construct"""
        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        scores = self.attention(x).squeeze()
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        expand_dims = ops.ExpandDims()
        weighted = ops.mul(x, expand_dims(scores, -1).expand_as(x))
        # weighted = mul(inputs, expand_dims(scores, -1).expand_as(inputs))

        # representations = weighted.sum(1).squeeze()
        op = ops.ReduceSum(keep_dims=True)
        representations = op(weighted, 1).squeeze()
        return representations, scores
