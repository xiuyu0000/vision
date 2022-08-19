# SqueezeNet

***

SqueezeNet is the name of a [deep neural network](https://en.wikipedia.org/wiki/Deep_neural_network) for [computer vision](https://en.wikipedia.org/wiki/Computer_vision) that was released in 2016. SqueezeNet was developed by researchers at [DeepScale](https://en.wikipedia.org/wiki/DeepScale), [University of California, Berkeley](https://en.wikipedia.org/wiki/University_of_California,_Berkeley), and [Stanford University](https://en.wikipedia.org/wiki/Stanford_University). In designing SqueezeNet, the authors' goal was to create a smaller neural network with fewer parameters that can more easily fit into computer memory and can more easily be transmitted over a computer network.

The architectural definition of each network refers to the following papers:

[1] Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer. [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360). arXiv preprint arXiv: 1602.07360, 2016.

## pretrained models

***

All resnet checkpoints were trained with image resolution 224x224. Each model verifies the accuracy

of Top-1 and Top-5, and compares it with that of pytorch and the paper. The table is as follows:

|  | | MindSpore | MindSpore | Pytorch | Pytorch | Paper | Paper | | |
|:-----:|:---------:|:--------:|:---------:|:---------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| Model | Dataset | Top-1 (%) | Top-5 (%) | Top-1 (%) | Top-5 (%) | Top-1 (%) | Top-5 (%) | Download | Config |
| Squeezenet_v0 | ImageNet | 51.94 | 75.34 | / | / | 57.5 | 80.3 | / | / |
| Squeezenet_v1 | ImageNet | 51.99 | 76.58 | / | / | 57.5 | 80.3 | / | / |

## Training

***

### Parameter description

| Parameter | Default | Description |
|:-----|:---------|:--------|
| model |  | Model Type |
| device_target | GPU | Hardware device |
| data_url |  | Path to training dataset |
| pretrained | False | Load pretrained model |
| run_distribute | True | Distributed parallel training |
| num_parallel_workers | 4               | Number of parallel workers |
| dataset_sink_mode | True | Data sinking mode |
| num_classes | 1000 | Number of dataset classifications |
| batch_size | 64 | Number of batch size |
| repeat_num | 1 | Number of data repetitions |
| momentum | 0.9 | Momentum parameter |
| epoch_size | 90 | Number of epoch |
| keep_checkpoint_max | 10 | Maximum number of checkpoints saved |
| ckpt_save_dir | './squeezenetnet' | Save path of checkpoint |
| decay_epoch | 90 | Number of decay epoch |
| lr_decay_mode | cosine_decay_lr | Learning rate decay mode |
| smooth_factor | 0.1 | Label smoothing factor |
| max_lr | 0.1 | maximum learning rate |
| min_lr | 0.0 | minimum learning rate |
| milestone |  | A list of milestone |
| learning_rates |  | A list of learning rates |
| resize | 224 | Resize the image |

## Examples

***

### Train

- The following configuration uses 2 GPUs for training.

  ```shell
  mpirun -n 2 python examples/classification/squeezenet/squeezenet_imagenet_train.py --model squeezenet_v0 --data_url ./dataset/imagenet --lr_decay_mode cosine_decay_lr --lr_max 0.05
  ```

  output:

  ```text
  Epoch:[0/90], step:[10009/10009], loss:[3.722/3.722], time:2833004.205 ms, lr:0.05000
  Epoch time: 2844470.698 ms, per step time: 284.191 ms, avg loss: 3.722
  Epoch:[0/90], step:[10009/10009], loss:[3.383/3.383], time:2832985.714 ms, lr:0.05000
  Epoch:[1/90], step:[10009/10009], loss:[3.412/3.412], time:2816048.004 ms, lr:0.04998
  Epoch time: 2816069.191 ms, per step time: 281.354 ms, avg loss: 3.412
  Epoch:[1/90], step:[10009/10009], loss:[3.652/3.652], time:2816092.433 ms, lr:0.04998
  Epoch time: 2816094.680 ms, per step time: 281.356 ms, avg loss: 3.652
  Epoch:[2/90], step:[10009/10009], loss:[3.252/3.252], time:2811130.920 ms, lr:0.04994
  Epoch time: 2811143.910 ms, per step time: 280.862 ms, avg loss: 3.252
  Epoch:[2/90], step:[10009/10009], loss:[3.166/3.166], time:2811259.489 ms, lr:0.04994
  Epoch time: 2811261.716 ms, per step time: 280.873 ms, avg loss: 3.166
  Epoch:[3/90], step:[10009/10009], loss:[3.400/3.400], time:5210716.454 ms, lr:0.04986
  Epoch time: 5210735.076 ms, per step time: 520.605 ms, avg loss: 3.400
  Epoch:[3/90], step:[10009/10009], loss:[3.463/3.463], time:5211032.215 ms, lr:0.04986
  Epoch time: 5211050.575 ms, per step time: 520.636 ms, avg loss: 3.463
  Epoch:[4/90], step:[10009/10009], loss:[3.447/3.447], time:5399162.229 ms, lr:0.04976
  Epoch time: 5399178.715 ms, per step time: 539.432 ms, avg loss: 3.447
  Epoch:[4/90], step:[10009/10009], loss:[3.263/3.263], time:5399334.003 ms, lr:0.04976
  Epoch time: 5399430.792 ms, per step time: 539.458 ms, avg loss: 3.263
  ...
  ```

### Eval

- The following configuration for eval.

  ```shell
  python examples/classification/squeezenet/squeezenet_imagenet_eval.py --model squeezenet_v0 --data_url ./dataset/imagenet --pretrained True
  ```

  output:

  ```text
  {'Top_1_Accuracy': 0.5194310897435898, 'Top_5_Accuracy': 0.7534455128205129}
  ```

### Infer

- The following configuration for infer.

  ```shell
  python examples/classification/squeezenet/squeezenet_imagenet_infer.py --model squeezenet_v0 --pretrained True --data_url ./infer
  ```
