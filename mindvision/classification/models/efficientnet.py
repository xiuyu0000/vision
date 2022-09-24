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
"""EfficientNet Architecture"""

from typing import Any

from mindvision.classification.models.head import DenseHead
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.models.backbones import EfficientNet

__all__ = [
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
]


def _efficientnet(arch: str,
                  width_mult: float,
                  depth_mult: float,
                  dropout: float,
                  input_channel: int,
                  num_classes: int,
                  pretrained: bool,
                  **kwargs: Any,
                  ) -> EfficientNet:
    """EfficientNet architecture."""

    backbone = EfficientNet(width_mult, depth_mult, **kwargs)
    head = DenseHead(input_channel, num_classes, keep_prob=1 - dropout)
    model = BaseClassifier(backbone, head=head)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        LoadPretrainedModel(model, model_urls[arch]).run()
    return model


def efficientnet_b0(num_classes: int = 1000,
                    pretrained: bool = False,
                    ) -> EfficientNet:
    """
    Constructs a EfficientNet B0 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_classes (int): The numbers of classes. Default: 1000.
        pretrained (bool): If True, returns a model pre-trained on IMAGENET. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import efficientnet_b0
        >>>
        >>> net = efficientnet_b0(1000, False)
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About EfficientNet:

    EfficientNet systematically studys model scaling and identify that carefully balancing network depth, width,
    and resolution can lead to better performance. Based on this observation, The model proposes a new scaling method
    that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound
    coefficient. This model demonstrates the effectiveness of this method on scaling up MobileNets and ResNet.

    Citation:

    .. code-block::

        @misc{tan2020efficientnet,
            title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
            author={Mingxing Tan and Quoc V. Le},
            year={2020},
            eprint={1905.11946},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }
    """
    return _efficientnet("efficientnet_b0", 1.0, 1.0, 0.2, 1280, num_classes, pretrained)


def efficientnet_b1(num_classes: int = 1000,
                    pretrained: bool = False,
                    ) -> EfficientNet:
    """
    Constructs a EfficientNet B1 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_classes (int): The numbers of classes. Default: 1000.
        pretrained (bool): If True, returns a model pre-trained on IMAGENET. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import efficientnet_b1
        >>>
        >>> net = efficientnet_b1(1000, False)
        >>> x = ms.Tensor(np.ones([1, 3, 240, 240]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About EfficientNet:

    EfficientNet systematically studys model scaling and identify that carefully balancing network depth, width,
    and resolution can lead to better performance. Based on this observation, The model proposes a new scaling method
    that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound
    coefficient. This model demonstrates the effectiveness of this method on scaling up MobileNets and ResNet.

    Citation:

    .. code-block::

        @misc{tan2020efficientnet,
            title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
            author={Mingxing Tan and Quoc V. Le},
            year={2020},
            eprint={1905.11946},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }
    """
    return _efficientnet("efficientnet_b1", 1.0, 1.1, 0.2, 1280, num_classes, pretrained)


def efficientnet_b2(num_classes: int = 1000,
                    pretrained: bool = False,
                    ) -> EfficientNet:
    """
    Constructs a EfficientNet B2 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_classes (int): The numbers of classes. Default: 1000.
        pretrained (bool): If True, returns a model pre-trained on IMAGENET. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import efficientnet_b2
        >>>
        >>> net = efficientnet_b2(1000, False)
        >>> x = ms.Tensor(np.ones([1, 3, 260, 260]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About EfficientNet:

    EfficientNet systematically studys model scaling and identify that carefully balancing network depth, width,
    and resolution can lead to better performance. Based on this observation, The model proposes a new scaling method
    that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound
    coefficient. This model demonstrates the effectiveness of this method on scaling up MobileNets and ResNet.

    Citation:

    .. code-block::

        @misc{tan2020efficientnet,
            title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
            author={Mingxing Tan and Quoc V. Le},
            year={2020},
            eprint={1905.11946},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }
    """
    return _efficientnet("efficientnet_b2", 1.1, 1.2, 0.3, 1408, num_classes, pretrained)


def efficientnet_b3(num_classes: int = 1000,
                    pretrained: bool = False,
                    ) -> EfficientNet:
    """
    Constructs a EfficientNet B3 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_classes (int): The numbers of classes. Default: 1000.
        pretrained (bool): If True, returns a model pre-trained on IMAGENET. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import efficientnet_b3
        >>>
        >>> net = efficientnet_b3(1000, False)
        >>> x = ms.Tensor(np.ones([1, 3, 300, 300]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About EfficientNet:

    EfficientNet systematically studys model scaling and identify that carefully balancing network depth, width,
    and resolution can lead to better performance. Based on this observation, The model proposes a new scaling method
    that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound
    coefficient. This model demonstrates the effectiveness of this method on scaling up MobileNets and ResNet.

    Citation:

    .. code-block::

        @misc{tan2020efficientnet,
            title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
            author={Mingxing Tan and Quoc V. Le},
            year={2020},
            eprint={1905.11946},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }
    """
    return _efficientnet("efficientnet_b3", 1.2, 1.4, 0.3, 1536, num_classes, pretrained)


def efficientnet_b4(num_classes: int = 1000,
                    pretrained: bool = False,
                    ) -> EfficientNet:
    """
    Constructs a EfficientNet B4 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_classes (int): The numbers of classes. Default: 1000.
        pretrained (bool): If True, returns a model pre-trained on IMAGENET. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import efficientnet_b4
        >>>
        >>> net = efficientnet_b4(1000, False)
        >>> x = ms.Tensor(np.ones([1, 3, 380, 380]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About EfficientNet:

    EfficientNet systematically studys model scaling and identify that carefully balancing network depth, width,
    and resolution can lead to better performance. Based on this observation, The model proposes a new scaling method
    that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound
    coefficient. This model demonstrates the effectiveness of this method on scaling up MobileNets and ResNet.

    Citation:

    .. code-block::

        @misc{tan2020efficientnet,
            title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
            author={Mingxing Tan and Quoc V. Le},
            year={2020},
            eprint={1905.11946},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }
    """
    return _efficientnet("efficientnet_b4", 1.4, 1.8, 0.4, 1792, num_classes, pretrained)


def efficientnet_b5(num_classes: int = 1000,
                    pretrained: bool = False,
                    ) -> EfficientNet:
    """
    Constructs a EfficientNet B5 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_classes (int): The numbers of classes. Default: 1000.
        pretrained (bool): If True, returns a model pre-trained on IMAGENET. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import efficientnet_b5
        >>>
        >>> net = efficientnet_b5(1000, False)
        >>> x = ms.Tensor(np.ones([1, 3, 456, 456]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About EfficientNet:

    EfficientNet systematically studys model scaling and identify that carefully balancing network depth, width,
    and resolution can lead to better performance. Based on this observation, The model proposes a new scaling method
    that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound
    coefficient. This model demonstrates the effectiveness of this method on scaling up MobileNets and ResNet.

    Citation:

    .. code-block::

        @misc{tan2020efficientnet,
            title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
            author={Mingxing Tan and Quoc V. Le},
            year={2020},
            eprint={1905.11946},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }
    """
    return _efficientnet(
        "efficientnet_b5",
        1.6,
        2.2,
        0.4,
        2048,
        num_classes,
        pretrained,
    )


def efficientnet_b6(num_classes: int = 1000,
                    pretrained: bool = False,
                    ) -> EfficientNet:
    """
    Constructs a EfficientNet B6 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_classes (int): The numbers of classes. Default: 1000.
        pretrained (bool): If True, returns a model pre-trained on IMAGENET. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import efficientnet_b4
        >>>
        >>> net = efficientnet_b4(1000, False)
        >>> x = ms.Tensor(np.ones([1, 3, 528, 528]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About EfficientNet:

    EfficientNet systematically studys model scaling and identify that carefully balancing network depth, width,
    and resolution can lead to better performance. Based on this observation, The model proposes a new scaling method
    that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound
    coefficient. This model demonstrates the effectiveness of this method on scaling up MobileNets and ResNet.

    Citation:

    .. code-block::

        @misc{tan2020efficientnet,
            title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
            author={Mingxing Tan and Quoc V. Le},
            year={2020},
            eprint={1905.11946},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }
    """
    return _efficientnet(
        "efficientnet_b6",
        1.8,
        2.6,
        0.5,
        2304,
        num_classes,
        pretrained,
    )


def efficientnet_b7(num_classes: int = 1000,
                    pretrained: bool = False,
                    ) -> EfficientNet:
    """
    Constructs a EfficientNet B7 architecture from
    `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

    Args:
        num_classes (int): The numbers of classes. Default: 1000.
        pretrained (bool): If True, returns a model pre-trained on IMAGENET. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import efficientnet_b7
        >>>
        >>> net = efficientnet_b7(1000, False)
        >>> x = ms.Tensor(np.ones([1, 3, 600, 600]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About EfficientNet:

    EfficientNet systematically studys model scaling and identify that carefully balancing network depth, width,
    and resolution can lead to better performance. Based on this observation, The model proposes a new scaling method
    that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound
    coefficient. This model demonstrates the effectiveness of this method on scaling up MobileNets and ResNet.

    Citation:

    .. code-block::

        @misc{tan2020efficientnet,
            title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
            author={Mingxing Tan and Quoc V. Le},
            year={2020},
            eprint={1905.11946},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }
    """
    return _efficientnet(
        "efficientnet_b7",
        2.0,
        3.1,
        0.5,
        2560,
        num_classes,
        pretrained,
    )
