# 配置文件 Config

MindSpore Vision整体采用模块化以及继承设计，所有的网络模型均可以采用配置进行组合设计，而且配置文件也可以各自继承，方便进行各种实验。当前Classification的配置文件采用yaml格式。

## 配置文件目录结构

配置文件由基础配置`configs/base`和各类模型配置`configs/yolo`组成；基础配置有两个基本类型：数据集和模型。在使用套件进行检测模型生成时可以选择基础配置文件组合成模型配置来进行模型训练。

目录结构如下，由基础配置和相关模型配置组成：

```text
|-- Configs
  |-- base
  |   |-- datasets
  |   |   |__yolo_coco_dataset.yaml
  |   |__ models
  |       |__yolov4_cspdarknet53.yaml
  |-- yolo
  |   |__ yolov4_cspdarknet53_coco.yaml
  ...
```

对于基础文件夹中同一个文件夹下的配置，建议只选择一个基础配置。

## 配置文件命名规则

我们遵循一下样式来命名配置文件。建议遵循相同的风格。

```text
{model}_{backbone}_[neck]_{parameter}_{dataset}.yaml
```

`{xxx}`是必填字段，`[yyy]`是可选字段。

- `{model}`：模型名称，例如`faster_rcnn`，`yolo`等。
- `{backbone}`：主干网络，如`resnet50`，`darknet53`等。
- [neck]：如`fpn`等。
- `{dataset}`：数据集如coco，voc等。

## 运行时修改配置

在使用`tools/train.py`或者`tools/eval.py`进行模型训练和验证时，可以指定`--options`就地修改配置。

- 更新字典值`value`。

  可以按照原始配置中的dict键的顺序指定配置选项进行值更新。例如，将模型的训练参数中的分布式训练开关打开：`--options model.train.is_distributed=True`。

- 更新字典中的列表/元组的值。

  如果要更新的值是列表或者元组，更新的方式和更新字典的值一样。例如更新学习率参数：`--options learning_rate.lr_epochs=[200, 250]`。

## 继承配置文件的修改

MindSpore Vision的配置文件是可以引用基础配置，若需要修改基础配置，只需要在引用的配置文件中去按照dict的键值顺序进行指定配置选项进行值更新。例如：

数据集的基础配置`configs/base/datasets/yolo_coco_dataset.yaml`：

```yaml
data_loader:
    train:
        dataset:
            type: GeneratorDataset
            source:
                type: COCOYoloDataset
                root: "/home/dataset/coco/train2017"
                ann_file: "/home/dataset/coco/annotations/instances_train2017.json"
            sampler:
                type: DistributedSampler
                num_replicas: 1
                rank: 0
                shuffle: True

            column_names: [ "image", "bbox1", "bbox2", "bbox3", "gt_box1", "gt_box2", "gt_box3" ]
            num_parallel_workers: 8
```

模型的基础配置`configs/base/models/yolo_cspdarknet53.yaml`：

```yaml
model:
    type: YOLOv4
    backbone:
        type: CspDarkNet
        in_channels: [32, 64, 128, 256, 512]
        out_channels: [64, 128, 256, 512, 1024]
        layer_nums: [1, 2, 8, 8, 4]
    neck:
        type: YOLOv4Neck
        backbone_shape: [64, 128, 256, 512, 1024]
        out_channel: 255 # 3 * (num_classes + 5)
    bbox_head:
        type: YOLOv4Head
        l_scale_x_y: 1.05
        l_offset_x_y: 0.025
        m_scale_x_y: 1.1
        m_offset_x_y: 0.05
        s_scale_x_y: 1.2
        s_offset_x_y: 0.1
```

整体配置`configs/yolo/yolov4_cspdarknet53_coco.yaml`，在下面的配置中引用了上述的两个基础配置，当需要修改相关的配置参数时，则只需要在本配置中修改相关的键值对，例如：

```yaml
base_config: ['../base/datasets/yolo_coco_dataset.yaml',
              '../base/models/yolov4_cspdarknet53.yaml']
model：
    bbox_head：
        l_scale_x_y: 1.02
        l_offset_x_y: 0.015
```

## 数据集配置说明

数据集配置主要分为两个主体部分：

- **一是dataset配置，配置dataset迭代器的生成，如下所示：**

```yaml
dataset: # Config of dataset generator
    type: GeneratorDataset  # Type of dataset, GeneratorDataset is MindSpore builtin dataset that generates data from                                       # Python by invoking Python data source each epoch.
    source: # Config of generator callable object.
        type: COCOYoloDataset # COCOYoloDataset is custom coco dataset generator
        root: "/home/dataset/coco/train2017"    # The root path of coco2017 training data.
        ann_file: "/home/dataset/coco/annotations/instances_train2017.json" # The file path of annotation file.
    sampler: # Config of dataset sampler that used to choose samples from the dataset.
        type: DistributedSampler    # Type of sampler
        num_replicas: 1             # The number of processes.
        rank: 0                     # Rank of the current process within num_replicas.
        shuffle: True               # If true, sampler will shuffle the indices.
    column_names: [ "image", "bbox1", "bbox2", "bbox3", "gt_box1", "gt_box2", "gt_box3" ]                                                           # List of column data types of the dataset (default=None). If provided, sanity check will be
                            # performed on generator output
    num_parallel_workers: 8 # Number of subprocesses used to fetch the dataset in parallel.
```

`dataset`：关键字表示是数据集产生的配置参数的标识。

`type`：表示采用的哪个Class产生数据集，例如上面的例子采用的是MindSpore中的`Class GeneratorDataset`；而`type`下面的键值对`（source、sampler、column_names、num_parallel_workers）`是`type`标识的这个类的构造函数的参数。

`source`：是用来配置用户自定义的数据集解析的参数，上述的配置中采用了自定义的`COCOYoloDataset`，而COCOYoloDataset构造后作为`GeneratorDataset`的参数。

`sampler`：数据采样的配置，用于配置自定义的采样器。

`column_names`：MindSpore中数据集最终都会处理为tuple类型，column_names就是tuple中每一个数据列的名称。

`num_parallel_workers`：需要多少个数据加载线程。

- **二是数据增强的配置，具体举例说明如下：**

方式一：每一个batch进行数据增强，此种方式调用的是MindSpore的dataset.batch()进行数据增强，下面的参数配置就是对函数dataset.batch()的入参的配置。

```yaml
batch:  # Config of MindSpore's function: dataset.batch().
    per_batch_map:  # Per batch map callable.
        type: PerBatchMap # The type of per_batch_map.
        out_orders: ["image", "bbox1", "bbox2", "bbox3", "gt_box1", "gt_box2", "gt_box3"]
        # The output orders of dataset tuple.
        multi_scales: [ [ 416, 416 ],
                        [ 448, 448 ],
                        [ 480, 480 ],
                        [ 512, 512 ],
                        [ 544, 544 ],
                        [ 576, 576 ],
                        [ 608, 608 ],
                        [ 640, 640 ],
                        [ 672, 672 ],
                        [ 704, 704 ],
                        [ 736, 736 ] ] # Resize scale parameters.
        pipeline:   # Config of data preprocessing pipeline.
            - type: PerBatchCocoCollect # Converting tuple list to dict.
            - type: ResizeWithinMultiScales # The type of transformers, resize image size and bboxes size.
              max_boxes: 90
              jitter: 0.3
              max_trial: 10
            - type: RandomPilFlip               # The type of transformers, filp image.
            - type: ConvertGrayToColor          # The type of transformers, convert gray image to colors
            - type: ColorDistortion             # The type of transformers, color distortion.
              hue: 0.1
              saturation: 1.5
              value: 1.5
            - type: StaticNormalize             # The type of transformers, normalize image.
            - type: Transpose                   # The type of transformers, transpose image.
            - type: YoloBboxPreprocess          # The type of transformers, classify bboxer for yolo.
              anchors: [[12, 16],
                        [19, 36],
                        [40, 28],
                        [36, 75],
                        [76, 55],
                        [72, 146],
                        [142, 110],
                        [192, 243],
                        [459, 401]]
              anchor_mask: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
              num_classes: 80
              label_smooth: 0
              label_smooth_factor: 0.1
              iou_threshold: 0.213
              max_boxes: 90

    input_columns: [ "image", "bbox1", "bbox2", "bbox3", "gt_box1", "gt_box2", "gt_box3" ]
    output_columns: [ "image", "bbox1", "bbox2", "bbox3", "gt_box1", "gt_box2", "gt_box3" ]
    num_parallel_workers: 8
    drop_remainder: True
    batch_size: 8
```

`per_batch_map`：对batch进行数据增强处理的参数配置，其中才用了`PerBatchMap`类进行数据增强操作，而在`PerBatchMap`的参数中配置了数据增强的pipeline，每一个操作均是Detection中的transformer的一个变换类。

`input_columns`：输入数据的列名称。

`output_columns`：输出数据的列名称。

`num_parallel_workers`：数据增强处理的线程数。

`drop_remainder`：是否丢弃最后多余的数据。

`batch_size`：每一个batch的大小。

方式二：对每一张图片进行数据增强，此时调用的是MindSpore的dataset.map()，具体配置如下。

```yaml
map:                                        # Config of MindSpore's function: dataset.map().
    operations:                             # Config of operations which is one argument of function dataset.map().
        - type: EvalFormat                  # The type of transformers.
        - type: PilResize                   # The type of transformers.
          resize_size: [608, 608]
        - type: StaticNormalize             # The type of transformers.
        - type: ConvertGrayToColor          # The type of transformers.
        - type: HWC2CHW                     # The type of transformers.
        - type: Collect                     # The type of transformers.
    output_orders: [ "image", "image_id", "image_shape" ]
    input_columns: [ "image", "image_id" ]
    output_columns: [ "image", "image_id", "image_shape" ]
    column_order: [ "image", "image_id", "image_shape" ]
    num_parallel_workers: 8
```

`operations`：数据增强的配置，其中配置了数据增强的流水线，类似于batch中配置的pipeline参数。

`output_orders`：数据增强后数据的排布顺序。

`input_columns`：输入数据的列名称。

`output_columns`：输出数据的列名称。

`column_order`：输出数据的排布顺序。

`num_parallel_workers`：数据增强处理的线程数。

完整的数据集配置示例，以Yolov4为例介绍，具体的数据增强变换参数配置参考相关APIs。

```yaml
data_loader:    # Config of data loader
    train:  # Config of training dataset
        dataset: # Config of dataset generator
            type: GeneratorDataset  # Type of dataset, GeneratorDataset is MindSpore builtin dataset that generates data from                                       # Python by invoking Python data source each epoch.
            source: # Config of generator callable object.
                type: COCOYoloDataset # COCOYoloDataset is custom coco dataset generator
                root: "/home/dataset/coco/train2017"    # The root path of coco2017 training data.
                ann_file: "/home/dataset/coco/annotations/instances_train2017.json" # The file path of annotation file.
            sampler: # Config of dataset sampler that used to choose samples from the dataset.
                type: DistributedSampler    # Type of sampler
                num_replicas: 1             # The number of processes.
                rank: 0                     # Rank of the current process within num_replicas.
                shuffle: True               # If true, sampler will shuffle the indices.

            column_names: [ "image", "bbox1", "bbox2", "bbox3", "gt_box1", "gt_box2", "gt_box3" ]                                                           # List of column data types of the dataset (default=None). If provided, sanity check will be
                            # performed on generator output
            num_parallel_workers: 8 # Number of subprocesses used to fetch the dataset in parallel.
        batch:  # Config of MindSpore's function: dataset.batch().
            per_batch_map:  # Per batch map callable.
                type: PerBatchMap # The type of per_batch_map.
                out_orders: ["image", "bbox1", "bbox2", "bbox3", "gt_box1", "gt_box2", "gt_box3"]
                            # The output orders of dataset tuple.
                multi_scales: [ [ 416, 416 ],
                              [ 448, 448 ],
                              [ 480, 480 ],
                              [ 512, 512 ],
                              [ 544, 544 ],
                              [ 576, 576 ],
                              [ 608, 608 ],
                              [ 640, 640 ],
                              [ 672, 672 ],
                              [ 704, 704 ],
                              [ 736, 736 ] ] # Resize scale parameters.
                pipeline:   # Config of data preprocessing pipeline.
                    - type: PerBatchCocoCollect # Converting tuple list to dict.
                    - type: ResizeWithinMultiScales # The type of transformers, resize image size and bboxes size.
                      max_boxes: 90
                      jitter: 0.3
                      max_trial: 10
                    - type: RandomPilFlip               # The type of transformers, filp image.
                    - type: ConvertGrayToColor          # The type of transformers, convert gray image to colors
                    - type: ColorDistortion             # The type of transformers, color distortion.
                      hue: 0.1
                      saturation: 1.5
                      value: 1.5
                    - type: StaticNormalize             # The type of transformers, normalize image.
                    - type: Transpose                   # The type of transformers, transpose image.
                    - type: YoloBboxPreprocess          # The type of transformers, classify bboxer for yolo.
                      anchors: [[12, 16],
                                [19, 36],
                                [40, 28],
                                [36, 75],
                                [76, 55],
                                [72, 146],
                                [142, 110],
                                [192, 243],
                                [459, 401]]
                      anchor_mask: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
                      num_classes: 80
                      label_smooth: 0
                      label_smooth_factor: 0.1
                      iou_threshold: 0.213
                      max_boxes: 90

            input_columns: [ "image", "bbox1", "bbox2", "bbox3", "gt_box1", "gt_box2", "gt_box3" ]
            output_columns: [ "image", "bbox1", "bbox2", "bbox3", "gt_box1", "gt_box2", "gt_box3" ]
            num_parallel_workers: 8
            drop_remainder: True
            batch_size: 8

    eval:                           # Config of eval.
        dataset:                    # The same as training process.
            type: GeneratorDataset
            source:
                type: COCOYoloDataset
                root: "/home/dataset/coco/val2017"
                ann_file: "/home/dataset/coco/annotations/instances_val2017.json"
                is_training: False
            sampler:
                type: DistributedSampler
                num_replicas: 1
                rank: 0
                shuffle: True

            column_names: [ "image", "image_id" ]
            num_parallel_workers: 8
        map:                                        # Config of MindSpore's function: dataset.map().
            operations:                             # Config of operations which is one argument of function dataset.map().
                - type: EvalFormat                  # The type of transformers.
                - type: PilResize                   # The type of transformers.
                  resize_size: [608, 608]
                - type: StaticNormalize             # The type of transformers.
                - type: ConvertGrayToColor          # The type of transformers.
                - type: HWC2CHW                     # The type of transformers.
                - type: Collect                     # The type of transformers.
                  output_orders: [ "image", "image_id", "image_shape" ]
            input_columns: [ "image", "image_id" ]
            output_columns: [ "image", "image_id", "image_shape" ]
            column_order: [ "image", "image_id", "image_shape" ]
            num_parallel_workers: 8

        batch:                                      # Config of MindSpore's function: dataset.batch().
            batch_size: 8
            drop_remainder: True

    thread_num: 0
    group_size: 1
```

模型配置分为模型配置、训练参数配置、验证参数配置、推理参数配置、模型导出配置、学习率配置以及优化器配置，以Yolov4为例，具体如下：

- **模型配置**

  ```yaml
  model: # Config of model.
      type: YOLOv4 # The type of model, Class Yolov4.
      backbone: # Config of backbone.
          type: CspDarkNet # The type of backbone, Class CspDarkNet.
          in_channels: [32, 64, 128, 256, 512]
          out_channels: [64, 128, 256, 512, 1024]
          layer_nums: [1, 2, 8, 8, 4]
      neck: # Config of neck.
          type: YOLOv4Neck # The type of neck, Class CspDarkNet.
          backbone_shape: [64, 128, 256, 512, 1024]
          out_channel: 255 # 3 * (num_classes + 5)
      bbox_head: # Config of head.
          type: YOLOv4Head # The type of dense head, Class YOLOv4Head.
          l_scale_x_y: 1.05
          l_offset_x_y: 0.025
          m_scale_x_y: 1.1
          m_offset_x_y: 0.05
          s_scale_x_y: 1.2
          s_offset_x_y: 0.1
          anchor_generator: # Config of anchor generator.
              type: YoloAnchorGenerator
              anchor_scales: [ [ 12, 16 ],
                               [ 19, 36 ],
                               [ 40, 28 ],
                               [ 36, 75 ],
                               [ 76, 55 ],
                               [ 72, 146 ],
                               [ 142, 110 ],
                               [ 192, 243 ],
                               [ 459, 401 ] ]
              anchor_mask: [ [6, 7, 8],
                             [3, 4, 5],
                             [0, 1, 2] ]

          loss_cls: # Config of classification loss.
              type: CrossEntropyLoss
              use_sigmoid: True
              reduction: "sum"

          loss_confidence: # Config of confidence.
              type: CrossEntropyLoss
              use_sigmoid: True
              reduction: "sum"

          num_classes: 80
          ignore_threshold: 0.7
          is_training: True
  ```

  `backbone`：主干网络配置。

  `neck`：颈部连接配置。

  `bbox_head`：检测头配置。

- **训练参数配置**

  ```yaml
  train:
      # Path for local
      load_path: ""
      device_target: "Ascend"
      pretrained_backbone: ""
      resume_yolov4: ""
      pretrained_checkpoint: ""
      filter_weight: False
      max_epoch: 320
      ckpt_path: "outputs/"
      ckpt_interval: -1
      rank_save_ckpt_flag: 1
      is_distributed: False
      rank: 0
      run_eval: False
      save_best_ckpt: True
      checkpoint_filter_list: [ 'feature_map.backblock0.conv6.weight', 'feature_map.backblock0.conv6.bias',
                                'feature_map.backblock1.conv6.weight', 'feature_map.backblock1.conv6.bias',
                                'feature_map.backblock2.conv6.weight', 'feature_map.backblock2.conv6.bias',
                                'feature_map.backblock3.conv6.weight', 'feature_map.backblock3.conv6.bias' ]
      context:
          mode: 0 #0--Graph Mode; 1--Pynative Mode
          enable_auto_mixed_precision: True
          device_target: "Ascend"
          save_graphs: False
          device_id: 0

      need_profiler: 0
      profiler:
          is_detail: True
          is_show_op_path: True
      parallel:
          parallel_mode: "data_parallel"
          gradients_mean: True
          device_num: 1
      ckpt:
          max_num: 10
      train_wrapper:
          type: TrainingWrapper
  ```

  训练参数主要配置context、ckpt、parallel等参数

- **验证参数配置**

  ```yaml
  eval:
      # Eval options
      pretrained: "outputs/2021-06-26_time_14_32_14/ckpt_0/0-1_7328.ckpt"
      log_path: "eval/"
      test_nms_thresh: 0.45
      test_img_shape: [ 608, 608 ]
      eval_ignore_threshold: 0.001
      batch_size: 1
      ann_val_file: "/home/dataset/coco/annotations/instances_val2017.json"
      device_target: "Ascend"
  ```

- **推理参数配置**

  TODO

- **模型导出配置**

  ```yaml
  export:
      # Export options
      device_target: "Ascend"
      device_id: 0
      ckpt_file: "outputs/2021-06-24_time_22_22_21/ckpt_0/0-9_43966.ckpt"
      file_name: "yolov4"
      file_format: "AIR"
  ```

- **学习率配置**

  ```yaml
  learning_rate:
      lr_scheduler: "cosine_annealing"
      lr: 0.012
      lr_epochs: [220, 250]
      lr_gamma: 0.1
      eta_min: 0.0
      t_max: 320
      max_epoch: 320
      warmup_epochs: 20
  ```

- **优化器配置**

  ```yaml
  optimizer:
      type: Momentum
      momentum: 0.9
      weight_decay: 0.0005
      loss_scale: 64
  ```

而将上述的部分合并到一个文件中就得到了我们Yolov4的参数配置，可以结合数据集配置来进行模型训练和推理验证。
