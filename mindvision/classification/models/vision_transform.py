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
"""Vision Transformer."""

from typing import Optional
import ml_collections as collections

from mindspore import nn

from mindvision.classification.models.backbones.vit import ViT
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.models.head import DenseHead, MultilayerDenseHead
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel

__all__ = [
    'vit_b_16',
    'vit_l_16',
    'vit_b_32',
    'vit_l_32',
]


def vit(image_size: int,
        input_channels: int,
        patch_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        num_classes: int,
        mlp_dim: int,
        dropout: float = 0.,
        attention_dropout: float = 0.,
        drop_path_dropout: float = 0.,
        activation: nn.Cell = nn.GELU,
        norm: nn.Cell = nn.LayerNorm,
        pool: str = 'cls',
        representation_size: Optional[int] = None,
        pretrained: bool = False,
        arch: str = None) -> ViT:
    """Vision Transformer architecture."""
    backbone = ViT(image_size=image_size,
                   input_channels=input_channels,
                   patch_size=patch_size,
                   embed_dim=embed_dim,
                   num_layers=num_layers,
                   num_heads=num_heads,
                   mlp_dim=mlp_dim,
                   keep_prob=1.0 - dropout,
                   attention_keep_prob=1.0 - attention_dropout,
                   drop_path_keep_prob=1.0 - drop_path_dropout,
                   activation=activation,
                   norm=norm,
                   pool=pool)
    if representation_size:
        head = MultilayerDenseHead(input_channel=embed_dim,
                                   num_classes=num_classes,
                                   mid_channel=[representation_size],
                                   activation=['tanh', None],
                                   keep_prob=[1.0, 1.0])
    else:
        head = DenseHead(input_channel=embed_dim,
                         num_classes=num_classes)

    model = BaseClassifier(backbone=backbone, head=head)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load ckpt file.
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model


def vit_b_16(num_classes: int = 1000,
             image_size: int = 224,
             has_logits: bool = False,
             pretrained: bool = False,
             drop_out: float = 0.0,
             attention_dropout: float = 0.0,
             drop_path_dropout: float = 0.0
             ) -> ViT:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        image_size (int): The input image size. Default: 224 for ImageNet.
        num_classes (int): The number of classification. Default: 1000.
        has_logits (bool): Whether has logits or not. Default: False.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        drop_out (float): The drop out rate. Default: 0.0.
        attention_dropout (float): The attention dropout rate. Default: 0.0.
        drop_path_dropout (float): The stochastic depth rate. Default: 0.0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import vit_b_16
        >>>
        >>> net = vit_b_16()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About ViT:

    Vision Transformer (ViT) shows that a pure transformer applied directly to sequences of image
    patches can perform very well on image classification tasks. When pre-trained on large amounts
    of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet,
    CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art
    convolutional networks while requiring substantially fewer computational resources to train.

    Citation:

    .. code-block::

        @article{2020An,
        title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
        author={Dosovitskiy, A. and Beyer, L. and Kolesnikov, A. and Weissenborn, D. and Houlsby, N.},
        year={2020},
        }
    """
    config = collections.ConfigDict()
    config.arch = "vit_b_16_" + str(image_size)
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 16
    config.embed_dim = 768
    config.mlp_dim = 3072
    config.num_heads = 12
    config.num_layers = 12
    config.dropout = drop_out
    config.attention_dropout = attention_dropout
    config.drop_path_dropout = drop_path_dropout
    config.pretrained = pretrained
    config.input_channels = 3
    config.pool = 'cls'
    config.representation_size = 768 if has_logits else None

    return vit(**config)


def vit_l_16(num_classes: int = 1000,
             image_size: int = 224,
             has_logits: bool = False,
             pretrained: bool = False,
             drop_out: float = 0.0,
             attention_dropout: float = 0.0,
             drop_path_dropout: float = 0.0
             ) -> ViT:
    """
    Constructs a vit_l_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        image_size (int): The input image size. Default: 224 for ImageNet.
        num_classes (int): The number of classification. Default: 1000.
        has_logits (bool): Whether has logits or not. Default: False.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        drop_out (float): The drop out rate. Default: 0.0.
        attention_dropout (float): The attention dropout rate. Default: 0.0.
        drop_path_dropout (float): The stochastic depth rate. Default: 0.0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import vit_l_16
        >>>
        >>> net = vit_l_16()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About ViT:

    Vision Transformer (ViT) shows that a pure transformer applied directly to sequences of image
    patches can perform very well on image classification tasks. When pre-trained on large amounts
    of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet,
    CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art
    convolutional networks while requiring substantially fewer computational resources to train.

    Citation:

    .. code-block::

        @article{2020An,
        title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
        author={Dosovitskiy, A. and Beyer, L. and Kolesnikov, A. and Weissenborn, D. and Houlsby, N.},
        year={2020},
        }
    """

    config = collections.ConfigDict()
    config.arch = 'vit_l_16_' + str(image_size)
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 16
    config.embed_dim = 1024
    config.mlp_dim = 4096
    config.num_heads = 16
    config.num_layers = 24
    config.dropout = drop_out
    config.attention_dropout = attention_dropout
    config.drop_path_dropout = drop_path_dropout
    config.input_channels = 3
    config.pool = 'cls'
    config.pretrained = pretrained
    config.representation_size = 1024 if has_logits else None

    return vit(**config)


def vit_b_32(num_classes: int = 1000,
             image_size: int = 224,
             has_logits: bool = False,
             pretrained: bool = False,
             drop_out: float = 0.0,
             attention_dropout: float = 0.0,
             drop_path_dropout: float = 0.0
             ) -> ViT:
    """
    Constructs a vit_b_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        image_size (int): The input image size. Default: 224 for ImageNet.
        num_classes (int): The number of classification. Default: 1000.
        has_logits (bool): Whether has logits or not. Default: False.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        drop_out (float): The drop out rate. Default: 0.0.
        attention_dropout (float): The attention dropout rate. Default: 0.0.
        drop_path_dropout (float): The stochastic depth rate. Default: 0.0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import vit_b_32
        >>>
        >>> net = vit_b_32()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About ViT:

    Vision Transformer (ViT) shows that a pure transformer applied directly to sequences of image
    patches can perform very well on image classification tasks. When pre-trained on large amounts
    of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet,
    CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art
    convolutional networks while requiring substantially fewer computational resources to train.

    Citation:

    .. code-block::

        @article{2020An,
        title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
        author={Dosovitskiy, A. and Beyer, L. and Kolesnikov, A. and Weissenborn, D. and Houlsby, N.},
        year={2020},
        }
    """
    config = collections.ConfigDict()
    config.arch = 'vit_b_32_' + str(image_size)
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 32
    config.embed_dim = 768
    config.mlp_dim = 3072
    config.num_heads = 12
    config.num_layers = 12
    config.dropout = drop_out
    config.attention_dropout = attention_dropout
    config.drop_path_dropout = drop_path_dropout
    config.pretrained = pretrained
    config.input_channels = 3
    config.pool = 'cls'
    config.representation_size = 768 if has_logits else None

    return vit(**config)


def vit_l_32(num_classes: int = 1000,
             image_size: int = 224,
             has_logits: bool = False,
             pretrained: bool = False,
             drop_out: float = 0.0,
             attention_dropout: float = 0.0,
             drop_path_dropout: float = 0.0
             ) -> ViT:
    """
    Constructs a vit_l_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        image_size (int): The input image size. Default: 224 for ImageNet.
        num_classes (int): The number of classification. Default: 1000.
        has_logits (bool): Whether has logits or not. Default: False.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        drop_out (float): The drop out rate. Default: 0.0.
        attention_dropout (float): The attention dropout rate. Default: 0.0.
        drop_path_dropout (float): The stochastic depth rate. Default: 0.0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import vit_l_32
        >>>
        >>> net = vit_l_32()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About ViT:

    Vision Transformer (ViT) shows that a pure transformer applied directly to sequences of image
    patches can perform very well on image classification tasks. When pre-trained on large amounts
    of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet,
    CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art
    convolutional networks while requiring substantially fewer computational resources to train.

    Citation:

    .. code-block::

        @article{2020An,
        title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
        author={Dosovitskiy, A. and Beyer, L. and Kolesnikov, A. and Weissenborn, D. and Houlsby, N.},
        year={2020},
        }
    """
    config = collections.ConfigDict()
    config.arch = 'vit_l_32_' + str(image_size)
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 32
    config.embed_dim = 1024
    config.mlp_dim = 4096
    config.num_heads = 16
    config.num_layers = 24
    config.dropout = drop_out
    config.attention_dropout = attention_dropout
    config.drop_path_dropout = drop_path_dropout
    config.pretrained = pretrained
    config.input_channels = 3
    config.pool = 'cls'
    config.representation_size = 1024 if has_logits else None

    return vit(**config)
