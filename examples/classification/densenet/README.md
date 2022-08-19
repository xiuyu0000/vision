# Densenet

DenseNet 是由黄高等在 2017 年 CVPR 会议上提 出 的，其 网 络 原 理 和 ResNet 相 似 ，区 别 在 于DenseNet 是把所有前面层的输出连接在一起作为下一层的输入，进行前向传播时达到了有效特征的传递并缓解了梯度消失的问题。 为了便于下采样，DenseNet将网络划分为多个紧密连接的密集块，并通过作卷积和池化层之间过渡层来改变特征映射的大小。

网络架构参考如下论文

[1] Huang, Gao, et al. "Densely connected convolutional networks." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2017.

## pertrained model

|                     |          | MindSpore |          |
| ------------------- | -------- | --------- | -------- |
| Model               | Dataset  | Top-1 (%) | Download |
| DenseNet-121 (k=32) | ImageNet | 68.5      |          |
| DenseNet-169 (k=32) | ImageNet | 69.2      |          |

## Training

### Parameter description

| Parameter            | Default            | Description                                     |
| -------------------- | ------------------ | ----------------------------------------------- |
| model                | None               | Densenet model type                             |
| device_target        | GPU                | device to run the training                      |
| data_url             | None               | dataset location                                |
| epoch_size           | 300                | number of epochs to train the model             |
| pretrained           | False              | load pretrained model                           |
| keep_checkpoint_max  | 10                 | number of checkpoint to save down               |
| ckpt_save_dir        | ./densenet         | location of training output                     |
| num_parallel_workers | 1                  | number of parallel worker                       |
| repeat_num           | 1                  | number of repetition                            |
| num_classes          | 1000               | number of classification classes                |
| min_lr               | 0.0                | the end learning rate                           |
| max_lr               | 0.1                | the max learning rate                           |
| decay_epoch          | 300                | number of decay epochs                          |
| milestone            | [150,225,300]      | a list of milestone for learning rate decay     |
| learning_rate        | [0.1, 0.01, 0.001] | a list of learning rate for learning rate decay |
| momentum             | 0.9                | momentum for moving average                     |
| dataset_sink_mode    | True               | dataset sink or not                             |
| run_distribute       | False              | distributed training                            |
| resize               | 224                | resize the image                                |

## Example

### Train

The following configuration uses single GPU for training.

```bash
python densenet_imagenet_train.py --model densenet121 --data_url ./dataset/imagenet
```

output

```bash
Epoch:[  0/300], step:[20018/20018], loss:[3.345/3.345], time:3398221.223 ms, lr:0.10000
Epoch time: 3408753.056 ms, per step time: 170.284 ms, avg loss: 3.345
Epoch:[  1/300], step:[20018/20018], loss:[2.741/2.741], time:3402508.923 ms, lr:0.10000
Epoch time: 3402508.923 ms, per step time: 169.973 ms, avg loss: 2.741
...
```

### Eval

- The following configuration for eval.

  ```bash
  python densenet_imagenet_train.py --model densenet121 --data_url ./dataset/imagenet --pretrained True
  ```

  output:

  ```bash
  {'Top_1_Accuracy': 0.6854166666666667, 'Top_5_Accuracy': 0.8841546474358974}
  ```

### Infer

- The following configuration for infer.

  ```bash
  python densenet_imagenet_train.py --model densenet121 --pretrained True --data_url ./infer
  ```

  output:

  ```bash
  {151: 'Chihuahua'}
  ```
