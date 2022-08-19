# MindSpore Vision Documents

## Prerequisites

- numpy 1.17+
- opencv-python 4.1+
- pytest 4.3+
- [mindspore](https://www.mindspore.cn/install) 1.2+
- ml_collection
- tqdm
- pillow

## Installation

### Prepare environment

- Create a conda virtual environment and activate it.

```shell
conda create -n mindvision python=3.7.5 -y
conda activate mindvision
```

- Install MindSpore

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.3.0/MindSpore/ascend/aarch64/mindspore_ascend-1.3.0-cp37-cp37m-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Install MindVision

- Clone the MindVision repository.

```shell
git clone https://gitee.com/mindspore/vision.git
cd vision
```

- Install

```shell
python setup.py install
```

### Verification

To verify whether MindVision and the required environment are installed correctly, we can run sample Python code to
initialize a classificer and run inference a demo image:

The image to infer:
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

The above code is supposed to run successfully upon you finish the installation.

## Get Start

This tutorial is mainly aimed at primary users. The content mainly includes network training using mindvision's classification.

### Basic Knowledge

As the name suggests, image classification is a pattern classification problem, which is the most basic task in computer vision. Its goal is to divide different images into different categories:

- train/val/test dataset represent the training set, verification set and test set of the model respectively.

    - training dataset: used to train the model and enable the model to recognize different types of features.

    - val dataset: the test set in the training process, which is convenient to view the training degree of the model in the training process.

    - test dataset: the test dataset used to evaluate the model results after the training model is completed.

- epoch

  The total number of rounds of model training iterations. An epoch is when the model passes through all samples of the training set.

  When the difference between test error rate and training error rate is small, the current number of iteration rounds can be considered appropriate.

  When the test error decreases first and then increases, it indicates that the number of iteration rounds is too large, and the number of iteration rounds needs to be reduced, otherwise over fitting is easy to occur.

- loss Function

  During training, measure the difference between model output (predicted value) and real value.

- Acc

  Indicates the proportion of samples with correct prediction in the total data.

### Data Processing

Enter the `mindvision / classification`.

```shell
cd mindvision/classification
```

Download and unzip the dataset.

The MNIST dataset used in our example is composed of 10 categories of 28 * 28 gray images. The training dataset contains 60000 images and the test dataset contains 10000 images.

Modify download options and data saving path in `config/classification/datasets/mnist.yaml` file.

There are multiple parameter configurations in the yaml file, where ` path` is the save path of the dataset and
`download` is the download option. Fill in and save your path and download option of the dataset.

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

Return to ``mindvision/classification``。

```shell
cd mindvision/classification
```

### Model Training

```shell
python tools/train.py -c configs/lenet/lenet.yaml
```

- `-c` The parameter is the path of the configuration file of the specified training. The specific super parameters of the training can be viewed in the `yaml` file.
- The parameter `epochs` in `yaml` file is set to 20, indicating that 20 epochs iterations are performed on the whole data set.

### Model Verification

```shell
python tools/eval.py -c configs/lenet/lenet.yaml
```

- `-c` The parameter is the path of the configuration file of the specified training. The specific super parameters of the training can be viewed in the `yaml` file.

### Model Valuation

```shell
python tools/eval.py -c configs/lenet/lenet.yaml
```

- `-c` The parameter is the path of the configuration file of the specified training. The specific super parameters of the training can be viewed in the `yaml` file.

### 模型导出

```shell
python tools/eval.py -c configs/lenet/lenet.yaml
```

- `-c` The parameter is the path of the configuration file of the specified training. The specific super parameters of the training can be viewed in the `yaml` file.
