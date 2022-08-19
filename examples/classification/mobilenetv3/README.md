# MobileNetV3

***

Besides using inverted residual structure where the input and output of the residual block are thin bottleneck layers and lightweight depthwise convolutions of MobileNetV2, MobileNetV3 combines NAS and NetAdapt algorithm that can be transplanted  to mobile devices like mobile phones, in order to deliver the next generation of high accuracy efficient neural network models to power on-device computer vision.

The architectural definition of each network refers to the following papers:

[1] Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al."Searching for mobilenetv3."In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324.2019.

## Pretrained models

***

All Mobilenetv3 checkpoints were trained with image resolution 224x224. Each model verifies the accuracy

of Top-1 and Top-5, and compares it with that of the paper. The table is as follows:

|                   |          | MindSpore | MindSpore | Pytorch   | Pytorch   | Paper     | Paper     |          |        |
| ----------------- | -------- | --------- | --------- | --------- | --------- | --------- | --------- | -------- | ------ |
| Model             | Dataset  | Top-1 (%) | Top-5 (%) | Top-1 (%) | Top-5 (%) | Top-1 (%) | Top-5 (%) | Download | Config |
| Mobilenetv3_small | ImageNet | 63.8      | 84.8      | /         | /         | 67.4      | /         | /        | /      |
| Mobilenetv3_large | ImageNet | 70.0      | 89.2      | /         | /         | 75.2      | /         | /        | /      |

## Training

***

### Parameter description

| Parameter            | Default          | Description                             |
| :------------------- | :--------------- | :-------------------------------------- |
| device_target        | GPU              | Hardware device                         |
| data_url             |                  | Path to training dataset                |
| pretrained           | False            | Path to pretrained model                |
| run_distribute       | True             | Distributed parallel training           |
| num_parallel_workers | 8                | Number of parallel workers              |
| dataset_sink_mode    | True             | Data sinking mode                       |
| num_classes          | 1000             | Number of dataset classifications       |
| batch_size           | 64               | Number of batch size                    |
| repeat_num           | 1                | Number of data repetitions              |
| momentum             | 0.9              | Momentum parameter                      |
| epoch_size           | 200              | Number of epoch                         |
| keep_checkpoint_max  | 10               | Maximum number of checkpoints saved     |
| ckpt_save_dir        | './mobilenet_v3' | Save path of checkpoint                 |
| lr_decay_mode        | cosine_decay_lr  | Learning rate decay mode                |
| decay_epoch          | 200              | Number of decay epoch                   |
| smooth_factor        | 0.1              | Label smoothing factor                  |
| max_lr               | 0.1              | maximum learning rate                   |
| min_lr               | 0.0              | minimum learning rate                   |
| milestone            |                  | A list of milestone                     |
| learning_rates       |                  | A list of learning rates                |
| resize               | 224              | Resize the height and weight of picture |

## Examples

***

### Train

- The following configuration uses 8 GPUs for training and the image input size is set to 224.

  ```shell
  mpirun -n 8 python mobilenet_v3_imagenet_train.py --model mobilenet_v3_small --data_url ./dataset/imagenet --epoch_size 120
  ```

  output:

  ```text
  Epoch:[ 90/ 120], step:[    1/ 2502], loss:[2.771/2.771], time:177.988 ms, lr:0.07329
  Epoch:[ 90/ 120], step:[    2/ 2502], loss:[2.183/2.183], time:201.013 ms, lr:0.07329
  Epoch:[ 90/ 120], step:[    3/ 2502], loss:[2.595/2.595], time:45.759 ms, lr:0.07329
  Epoch:[ 90/ 120], step:[    4/ 2502], loss:[2.655/2.713], time:131.621 ms, lr:0.07329
  Epoch:[ 90/ 120], step:[    5/ 2502], loss:[2.767/2.475], time:133.700 ms, lr:0.07329
  Epoch:[ 90/ 120], step:[    6/ 2502], loss:[2.461/2.528], time:49.254 ms, lr:0.07329
  Epoch:[ 90/ 120], step:[    7/ 2502], loss:[2.469/2.508], time:55.803 ms, lr:0.07329
  ...
  ```

### Eval

- The following configuration for Mobilenet_v3_small eval.

  ```shell
  python mobilenet_v3_imagenet_eval.py --model mobilenet_v3_small --data_url ./dataset/imagenet
  ```

  output:

  ```text
  {'Top_1_Accuracy': 0.6381241997439181, 'Top_5_Accuracy': 0.8480513764404609}
  ```

### Infer

- The following configuration for infer. The image input size is set to 224.

  ```shell
  python mobilenet_v3_imagenet_infer.py --model mobilenet_v3_small --data_url ./infer
  ```