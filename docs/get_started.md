# MindSpore Vision 文档

## 环境依赖

- numpy 1.17+
- opencv-python 4.1+
- pytest 4.3+
- [mindspore](https://www.mindspore.cn/install) 1.2+
- ml_collection
- tqdm
- pillow

## 安装

### 环境准备

- 创建一个conda虚拟环境并且激活。

```shell
conda create -n mindvision python=3.7.5 -y
conda activate mindvision
```

- 安装MindSpore

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.3.0/MindSpore/ascend/aarch64/mindspore_ascend-1.3.0-cp37-cp37m-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 安装MindVision

- 使用git克隆MindVsion仓库。

```shell
git clone https://gitee.com/mindspore/vision.git
cd vision
```

- 安装

```shell
python setup.py install
```

### 验证

为了验证MindVision和所需的环境是否正确安装，我们可以运行示例代码来初始化一个分类器然后推理一张图片：

推理所用的图片:
![four](../tests/st/classification/dataset/mnist/mnist.jpg)

```shell
python ./examples/classification/lenet/lenet_mnist_infer.py \
        --data_url ./tests/st/classification/dataset/mnist/mnist.jpg \
        --pretrained True \
        --device_target CPU
```

```text
   {4: 'four'}
```

如果您成功安装，以上代码应该会成功运行。

## 快速入门

此教程主要针对初级用户，内容主要包括利用MindVision的classification进行网络训练。

### 基础知识

图像分类顾名思义就是一个模式分类问题，是计算机视觉中最基础的任务，它的目标是将不同的图像，划分到不同的类别：

- train/val/test dataset分别代表模型的训练集、验证集和测试集

    - 训练集（train dataset）：用来训练模型，使模型能够识别不同类型的特征
    - 验证集（val dataset）：训练过程中的测试集，方便训练过程中查看模型训练程度
    - 测试集（test dataset）：训练模型结束后，用于评价模型结果的测试集

- 迭代轮数（epoch）

  模型训练迭代的总轮数，模型对训练集全部样本过一遍即为一个epoch。
  当测试错误率和训练错误率相差较小时，可认为当前迭代轮数合适；
  当测试错误率先变小后变大时，则说明迭代轮数过大，需要减小迭代轮数，否则容易出现过拟合。

- 损失函数（Loss Function）

  训练过程中，衡量模型输出（预测值）与真实值之间的差异

- 准确率（Acc）

  表示预测正确的样本数占总数据的比例

### 数据的准备与处理

进入`mindvision/classification`目录。

```shell
cd mindvision/classification
```

下载并解压数据集。

我们示例中用到的MNIST数据集是由10类28∗28的灰度图片组成，训练数据集包含60000张图片，测试数据集包含10000张图片。

修改`config/classification/datasets/mnist.yaml`文件中的下载选项和数据保存路径。

yaml文件中有多个参数配置，其中`path`是数据集保存路径，`download`是下载选项，将你的保存数据集的路径和下载选项填入并保存。

```text
train:
  dataset:
    type: Mnist
    path: "yourpath" # Root directory of the Mnist dataset or inference image. This default path is for ST test.
    split: "train"
    batch_size: 32 # Batch size of dataset.
    repeat_num: 1 # The repeat num of dataset.
    shuffle: True # Perform shuffle on the dataset.
    num_parallel_workers: 1 # Number of subprocess used to fetch the dataset in parallel.
    download: True # Whether to download the dataset.

eval:
  dataset:
    type: Mnist
    path: "yourpath" # Root directory of the Mnist dataset or inference image. This default path is for ST test.
    split: "test"
    batch_size: 32
    num_parallel_workers: 1
    download: True
```

返回``mindvision/classification``根目录。

```shell
cd mindvision/classification
```

### 模型训练

```shell
python tools/train.py -c configs/lenet/lenet.yaml
```

- `-c` 参数是指定训练的配置文件路径，训练的具体超参数可查看`yaml`文件
- `yaml`文件中`epochs`参数设置为20，说明对整个数据集进行20个epoch迭代

### 模型验证

```shell
python tools/eval.py -c configs/lenet/lenet.yaml
```

- `-c` 参数是指定训练的配置文件路径，训练的具体超参数可查看`yaml`文件

### 模型推理

```shell
python tools/eval.py -c configs/lenet/lenet.yaml
```

- `-c` 参数是指定训练的配置文件路径，训练的具体超参数可查看`yaml`文件

### 模型导出

```shell
python tools/eval.py -c configs/lenet/lenet.yaml
```

- `-c` 参数是指定训练的配置文件路径，训练的具体超参数可查看`yaml`文件
